package nn.layers.ffn.moe;

import nn.core.Module;
import nn.core.Parameter;
import nn.core.activation.Softplus;
import nn.core.initializer.Initializer;
import nn.core.optimizer.Optimizer;
import nn.layers.Linear;
import org.nd4j.linalg.api.ops.impl.shape.OneHot;
import org.nd4j.linalg.factory.Nd4j;
import tensor.Tensor;

import java.util.List;

public class MoERouter implements Module {

    private final Linear projLogits;
    private final Linear projNoise;

    private final int d_model;
    private final int numExperts;
    private final int topK;
    private final boolean useNoisyTopK;
    private final double capacityFactor;

    private int expCapacity;

    // Caches
    private Tensor dispatch, combine;

    public Tensor getfMean() {
        return fMean;
    }

    public Tensor getpMean() {
        return pMean;
    }

    public Tensor getLogits() {
        return logits;
    }

    public Tensor getProbsAll() {
        return probsAll;
    }

    public Tensor getLogSumExpLogits() {
        return logSumExpLogits;
    }

    // For LB-Loss
    private Tensor pMean, fMean;
    // For Router-Z-Loss
    private Tensor logits, probsAll, logSumExpLogits;
    // Shapes
    private int B, T;


    private final boolean isTrainable;

    private Tensor X;

    public MoERouter(int d_model, int numExperts, int topK, boolean useNoisyTopK, double capacityFactor, Initializer wInit, boolean isTrainable) {
        this.d_model = d_model;
        this.numExperts = numExperts;
        this.topK = topK;
        this.useNoisyTopK = useNoisyTopK;
        this.capacityFactor = capacityFactor;

        this.isTrainable = isTrainable;

        this.projLogits = new Linear(d_model, numExperts, wInit, isTrainable);
        if(useNoisyTopK) this.projNoise = new Linear(d_model, numExperts, wInit, isTrainable);
        else this.projNoise = null;
    }

    @Override
    public Tensor forward(Tensor X, boolean training) {
        // Reshape : [B, T, d] -> [B*T, d]
        this.B = X.size(0);
        this.T = X.size(1);
        int numTokens = B * T;
        X = X.reshape(numTokens, d_model);
        this.X = X;

        //System.out.println("Flattened X [B*T, d] : \n" + X);

        // Compute logits for after softmax
        this.logits = projLogits.forward(X, training);        // [B*T, numExperts]
        // For Noisy Top-K routing
        if(useNoisyTopK && training) {
            Tensor noise = new Softplus().forward(projNoise.forward(X, true));
            noise = noise.mul(Tensor.randn(noise.shape()));
            logits = logits.add(noise);
        }

        //System.out.println("Router logit : \n" + logits);

        // Top-K expert selection : compute probs over active experts
        Tensor[] topKOut = logits.topK(topK, -1, true, true);
        Tensor topKLogits = topKOut[0];     // [B*T, topK]
        Tensor topKIndices = topKOut[1];
        // Softmax : [B*T, numExperts], set non-top-k vals to -inf
        Tensor probs = Tensor.fill(Float.NEGATIVE_INFINITY, logits.shape());
        for(int i=0;i<numTokens;i++) {
            for(int j=0;j<topK;j++) {
                int validIndex = topKIndices.getInt(i, j);
                double topKval = topKLogits.getDouble(i, j);

                probs.set(topKval, i, validIndex);
            }
        }
        //System.out.println("Top-" + topK + " Logits : \n" + topKLogits);
        //System.out.println("Top-" + topK + " Indices : \n" + topKIndices);
        probs = probs.softmax();
        //System.out.println("Softmax probabilities : \n" + probs);

        // Expert capacity = (capacityFactor * topK * (B*T) / numExperts)
        this.expCapacity = (int) Math.floor(capacityFactor * topK * numTokens / numExperts);
        expCapacity += expCapacity % 2; // make sure expCapacity is an even int

        // One-hot mask for chosen experts : 0 -> not chosen, 1 -> chosen
        // [B*T, topK, numExperts]
        Tensor expMask = new Tensor(Nd4j.exec(new OneHot(topKIndices.getNDArray(), numExperts))[0]);
        // [K, B*T, N]
        expMask = expMask.permute(2, 1, 0);
        //System.out.println("One-hot selection Mask (Original : [K, B*T, N], Print : [B*T, K, N] : \n" + expMask.permute(1, 0, 2));

        // Expert-wise cumsum to check whether the number of designated tokens is less than expCapacity (For all batches)
        // [K*B*T, N]
        Tensor expRank = expMask.reshape(topK * numTokens, numExperts);
        expRank = new Tensor(Nd4j.cumsum(expRank.getNDArray(), 0).sub(1.0));

        //System.out.println("cumsum : \n" + expRank);
        expRank = expRank.reshape(topK, numTokens, numExperts);

        // if less than expCapacity, apply mask
        Tensor keep = expRank.lt(expCapacity);
        expMask = expMask.mul(keep);
        //System.out.println("newly applied mask (less than capacity) :\n" + expMask);

        // Slot index one-hot (including expert axis)
        // rankSel : Slot index for each tokens in each expert
        Tensor rankSel = expRank.mul(expMask);
        //System.out.println("rankSel : Selected index for each expert : \n" + rankSel);
        // expRankSC : One-hot
        Tensor expRankSC = new Tensor(Nd4j.exec(new OneHot(rankSel.getNDArray(), expCapacity, -1, 1.0, 0.0))[0]);
        //System.out.println("expRankSC : One-hot of rankSel, unselected -> 0 : \n" + expRankSC);

        // Dispatch : token -> (expert, slot) 0/1 mask => "Where to go?"
        // [B*T, N, C]
        this.dispatch = expMask.reshape(topK, numTokens, numExperts, 1)
                .mul(expRankSC.reshape(topK, numTokens, numExperts, expCapacity))
                .sum(0, false);

        //System.out.println("dispatch : \n" + dispatch);

        // Combine : weights for weighted-sum
        // flatten softmax probs [B, T, N] -> [1, B*T, N], mul with expMask (broadcast)
        Tensor flatProb = probs.reshape(1, numTokens, numExperts);
        // [K, B*T, N]
        Tensor expWeights = expMask.mul(flatProb);
        // combine : [B*T, N, C], each assigned expert's gated weight for tokens
        this.combine = expWeights.reshape(topK, numTokens, numExperts, 1)
                .mul(expRankSC.reshape(topK, numTokens, numExperts, expCapacity))
                .sum(0, false);

        //System.out.println("combine : \n" + combine);

        // For LB-Loss
        // [N]
        this.fMean = dispatch.sum(0, false).sum(-1, false).div(numTokens * topK);
        // [N]
        this.pMean = probs.reshape(B, T, numExperts).mean(0, false).mean(0, false);

        // For Router-Z-Loss
        // Caching Pre-Topk softmax(logits)
        this.probsAll = logits.softmax();
        this.logSumExpLogits = logits.exp().sum(-1, true).log();

        // Identity return
        return X;
    }

    @Override
    public Tensor calcGradients(Tensor dY, boolean accumulate, double scale) {

        // Use LB-Loss, Router-Z-Loss to update projLogits, projNoise
        // dY : Auxiliary loss from LoadBalancingLoss, RouterZLoss
        projLogits.calcGradients(dY, accumulate, scale);        // accumulate will be true for 2 Losses (LB, Z loss)

        // Identity return (nothing to return)
        return dY;
    }

    @Override
    public void update(Optimizer optimizer) {
        projLogits.update(optimizer);
    }

    @Override
    public List<Parameter> parameters() {
        if(useNoisyTopK) {
            return List.of(projLogits.parameters().get(0), projNoise.parameters().get(0));
        }
        return List.of(projLogits.parameters().get(0));
    }

    @Override
    public void zeroGrad() {
        if(useNoisyTopK) {
            projNoise.zeroGrad();
        }
        projLogits.zeroGrad();
    }

    public int getNumExperts() {
        return numExperts;
    }

    public int getTopK() {
        return topK;
    }

    public Tensor getDispatch() {
        return dispatch;
    }

    public Tensor getCombine() {
        return combine;
    }

    public int getExpCapacity() {
        return expCapacity;
    }

    public Linear getProjLogits() {
        return projLogits;
    }
}
