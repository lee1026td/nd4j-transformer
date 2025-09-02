package nn.transformer.modules;

import nn.core.Module;
import nn.core.Parameter;
import nn.optimizer.Optimizer;
import nn.transformer.attention.MultiHeadAttention;
import tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

public class EncoderBlock implements Module {

    private final Module norm1, norm2;
    private final MultiHeadAttention mha;
    private final FeedForwardNetwork ffn;
    private final boolean isTrainable;

    public EncoderBlock(Module norm1,
                        Module norm2,
                        MultiHeadAttention mha,
                        FeedForwardNetwork ffn,
                        boolean isTrainable) {
        this.norm1 = norm1; this.norm2 = norm2;
        this.mha = mha; this.ffn = ffn;
        this.isTrainable = isTrainable;
    }

    @Override
    public Tensor forward(Tensor X, boolean training) {
        throw new UnsupportedOperationException("EncoderBlock requires (X, mask)");
    }

    @Override
    public Tensor forwardMany(boolean training, Tensor... xs) {
        if(xs == null || xs.length != 2) throw new IllegalArgumentException();
        Tensor X = xs[0]; Tensor mask = xs[1];

        if(mask == null) throw new IllegalArgumentException();

        // Pre-Norm Self-Attn
        Tensor N1 = norm1.forward(X, training);
        Tensor selfAttn = mha.forwardMany(training, N1, N1, mask);
        // Residual connection
        Tensor Y = X.add(selfAttn);

        // Pre-Norm FFN
        Tensor N2 = norm2.forward(Y, training);
        Tensor feedForward = ffn.forward(N2, training);
        // Residual connection
        Tensor Z = Y.add(feedForward);

        return Z;
    }

    @Override
    public Tensor calcGradients(Tensor dY, boolean accumulate, double scale) {
        Tensor dY_s = (scale == 1.0) ? dY : dY.mul(scale);

        // Z = Y + ffn(norm2(Y))
        Tensor dY1 = ffn.calcGradients(dY_s, accumulate, 1.0);
        Tensor dNorm2 = norm2.calcGradients(dY1, accumulate, 1.0);
        Tensor dR1 = dY1.add(dNorm2);

        // Y = X + selfAttn(norm1(X))
        Tensor[] gradSelfAttn = mha.calcGradientsMany(dR1, accumulate, 1.0);
        Tensor dX1 = gradSelfAttn[0];
        Tensor dNorm1 = norm1.calcGradients(dX1, accumulate, 1.0);

        return dR1.add(dNorm1);
    }

    @Override
    public void update(Optimizer optimizer) {
        if(isTrainable) {
            norm1.update(optimizer); norm2.update(optimizer);
            mha.update(optimizer);
            ffn.update(optimizer);
        }
    }

    @Override
    public List<Parameter> parameters() {
        List<Parameter> ps = new ArrayList<>();
        ps.addAll(norm1.parameters()); ps.addAll(norm2.parameters());
        ps.addAll(mha.parameters());
        ps.addAll(ffn.parameters());

        return ps;
    }

    @Override
    public void zeroGrad() {
        norm1.zeroGrad(); norm2.zeroGrad();
        mha.zeroGrad();
        ffn.zeroGrad();
    }
}
