package nn.transformer.attention;

import nn.core.Module;
import nn.core.Parameter;
import nn.initializer.Initializer;
import nn.layers.Linear;
import nn.optimizer.Optimizer;
import nn.transformer.mask.MaskUtils;
import tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

public class MultiHeadAttention implements Module {

    private final int d_model;
    private final int numHeads;
    private final boolean isTrainable;
    private final double attnDropProb, outDropProb;
    private final boolean isCross;

    // Self-Attention
    private Linear Wqkv;
    // Cross-Attention
    private Linear Wq, Wkv, Wo;
    private double attnScale;

    private Tensor Xq, Xkv, Qh, Kh, Vh, P, O;
    private Tensor mask;
    private Tensor attnDrop, outDrop;

    public MultiHeadAttention(int d_model, int numHeads,
                              Initializer init,
                              boolean isTrainable,
                              double attnDropProb,
                              double outDropProb, boolean isCross) {
        this.d_model = d_model;
        this.numHeads = numHeads;
        this.isTrainable = isTrainable;
        this.attnDropProb = attnDropProb;
        this.outDropProb = outDropProb;
        this.isCross = isCross;

        this.attnScale = Math.sqrt((double) d_model / numHeads);

        if(isCross) {
            this.Wq = new Linear(d_model, d_model, init, isTrainable);
            this.Wkv = new Linear(d_model, 2 * d_model, init, isTrainable);
        } else {
            this.Wqkv = new Linear(d_model, 3 * d_model, init, isTrainable);
        }

        this.Wo = new Linear(d_model, d_model, init, isTrainable);
    }

    // Self-Attention : Xq == Xkv (w/o mask)
    @Override
    public Tensor forward(Tensor X, boolean training) {
        throw new IllegalArgumentException("Self MHA expects (X, mask)");
    }

    // Self-Attention w/ mask
    public Tensor forward(Tensor X, Tensor mask, boolean training) {
        return forwardMany(training, X, X, mask);
    }

    // Cross-Attention w/ mask
    public Tensor forward(Tensor Xq, Tensor Xkv, Tensor Mask, boolean training) {
        return forwardMany(training, Xq, Xkv, Mask);
    }

    @Override
    public Tensor forwardMany(boolean training, Tensor... xs) {
        if(xs == null || xs.length != 3)
            throw new IllegalArgumentException("MHA expects (Xq, Xkv, mask), got : " + xs.length);

        this.Xq = xs[0];      // [B, Tq, d]
        this.Xkv = xs[1];    // [B, Tkv, d]
        this.mask = xs[2];

        // Cross-Attention Q, K, V
        Tensor Q, K, V;
        if(isCross) {
            Tensor KV = Wkv.forward(Xkv, training);
            Q = Wq.forward(Xq, training);
            K = KV.slice(-1, 0, d_model);
            V = KV.slice(-1, d_model, 2 * d_model);
        }
        // Self-Attention Q, K, V
        else {
            // Xq == Xkv
            Tensor X = Xq;
            Tensor QKV = Wqkv.forward(X, training);
            Q = QKV.slice(-1, 0, d_model);
            K = QKV.slice(-1, d_model, 2 * d_model);
            V = QKV.slice(-1, 2 * d_model, 3 * d_model);
        }

        // Split heads Q, K, V into h heads : [B, h, T, d_h], where d_h = d_model / h
        this.Qh = AttentionOps.splitHeads(Q, numHeads);
        this.Kh = AttentionOps.splitHeads(K, numHeads);
        this.Vh = AttentionOps.splitHeads(V, numHeads);

        /* Compte attention by each head */
        // Attention scores = QK^T/sqrt(d_k), d_k = d_model / numHeads
        Tensor attnScores = AttentionOps.qkScores(Qh, Kh, attnScale);

        // Apply attention mask (additive mask) + softmax (over last axis : -1) = logits
        Tensor maskApplied = AttentionOps.applyMask(attnScores, mask);
        this.P = AttentionOps.softmaxLast(maskApplied);     // [B, H, Tq, Tkv]

        // Applying attention dropout
        if(training) {
            this.attnDrop = MaskUtils.dropoutMaskLike(P, attnDropProb);
            this.P = P.mul(attnDrop);
        }
        else this.attnDrop = null;

        // Attention(Q, K, V) : P * V
        // [B, H, Tq, Tkv] x [B, H, Tkv, d_h] = [B, H, Tq, d_h]
        Tensor Oh = P.matmul(Vh);

        // Concat : H * [B, H, Tq, d_h] -> [B, Tq, H * d_vh]
        this.O = AttentionOps.mergeHeads(Oh);

        // Applying Linear projection with Wo, Y = O x Wo : [B, Tq, d_model]
        Tensor Y = Wo.forward(O, training);

        // Applying Final dropout
        if(training) {
            outDrop = MaskUtils.dropoutMaskLike(Y, outDropProb);
            Y = Y.mul(outDrop);
        } else outDrop = null;

        return Y;
    }

    @Override
    public Tensor[] calcGradientsMany(Tensor dY, boolean accumulate, double scale) {

        // Final dropout
        if(outDrop != null) dY = dY.mul(outDrop);

        // dY : [B, Tq, d_model] -> dO = dY x Wo^T
        Tensor dO = Wo.calcGradients(dY, accumulate, scale);

        // Split gradients by H -> into each head
        Tensor dOh = AttentionOps.splitHeads(dO, numHeads); // [B, H, Tq, d_h]

        // Oh = P x Vh
        // dP = dOh x Vh^T, dVh = P^T x dOh
        Tensor dP = dOh.matmul(Vh.transpose(-2, -1));   // [B, H, Tq, Tkv]
        Tensor dVh = P.transpose(-2, -1).matmul(dOh);   // [B, H, Tkv, d_h]

        // Attention dropout
        if(attnDrop != null) dP = dP.mul(attnDrop);

        // P = softmax(attnScores)
        // dScore = (dP - sum(dP * P, axis=-1)) * P
        Tensor sum = dP.mul(P).sum(-1, true);   // [B, H, Tq, 1]
        Tensor dScore = dP.sub(sum).mul(P);                  // [B, H, Tq, Tkv]

        // AttentionScore = (Qh x Kh^T)/attnScale + mask
        // dQh = (dScore x Kh)/attnScale, dKh = (dScore^T x Qh)/attnScale
        Tensor dQh = dScore.matmul(Kh).div(attnScale);                      // [B, H, Tq, d_h]
        Tensor dKh = dScore.transpose(-2, -1).matmul(Qh).div(attnScale);    // [B, H, Tkv, d_h]

        // Merge heads
        // dQ : [B, Tq, d_model], dK, dV : [B, Tkv, d_model]
        Tensor dQ = AttentionOps.mergeHeads(dQh);
        Tensor dK = AttentionOps.mergeHeads(dKh);
        Tensor dV = AttentionOps.mergeHeads(dVh);

        if(isCross) {
            // Cross-attention
            // Q = Xq x Wq, [K;V] = Xkv x Wkv

            // dXq
            Tensor dXq = Wq.calcGradients(dQ, accumulate, scale);
            // dXkv
            Tensor dKV = Tensor.concat(-1, dK, dV);     // [B, Tkv, 2*d_model]
            Tensor dXkv = Wkv.calcGradients(dKV, accumulate, scale);

            return new Tensor[]{ dXq, dXkv };
        } else {
            // Self-attention
            // [Q;K;V] = X x Wqkv, Wqkv : [B, T, 3 * d_model]
            Tensor dQKV = Tensor.concat(-1, dQ, dK, dV);
            Tensor dX = Wqkv.calcGradients(dQKV, accumulate, scale);

            return new Tensor[]{ dX, null };
        }
    }

    @Override
    public Tensor calcGradients(Tensor dY, boolean accumulate, double scale) {
        throw new UnsupportedOperationException("Use calcGradientsMany()");
    }

    @Override
    public void update(Optimizer optimizer) {
        if(isTrainable) {
            if(isCross) {
                Wq.update(optimizer);
                Wkv.update(optimizer);
            } else Wqkv.update(optimizer);

            Wo.update(optimizer);
        }
    }

    @Override
    public List<Parameter> parameters() {
        List<Parameter> ps = new ArrayList<>();
        if(isCross) {
            ps.addAll(Wq.parameters());
            ps.addAll(Wkv.parameters());
        }
        else ps.addAll(Wqkv.parameters());

        ps.addAll(Wo.parameters());

        return ps;
    }

    @Override
    public void zeroGrad() {
        if(isCross) {
            Wq.zeroGrad(); Wkv.zeroGrad();
        } else Wqkv.zeroGrad();

        Wo.zeroGrad();
    }
}
