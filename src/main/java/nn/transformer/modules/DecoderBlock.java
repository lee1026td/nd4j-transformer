package nn.transformer.modules;

import nn.core.Module;
import nn.core.Parameter;
import nn.optimizer.Optimizer;
import nn.transformer.attention.MultiHeadAttention;
import tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

public class DecoderBlock implements Module {

    private final Module norm1, norm2, norm3;
    private final MultiHeadAttention selfMHA, crossMHA;
    private final FeedForwardNetwork ffn;
    private final boolean isTrainable;

    public DecoderBlock(Module norm1,
                        Module norm2,
                        Module norm3,
                        MultiHeadAttention selfMHA,
                        MultiHeadAttention crossMHA,
                        FeedForwardNetwork ffn,
                        boolean isTrainable) {
        this.norm1 = norm1; this.norm2 = norm2; this.norm3 = norm3;
        this.selfMHA = selfMHA; this.crossMHA = crossMHA;
        this.ffn = ffn;
        this.isTrainable = isTrainable;
    }

    @Override
    public Tensor forward(Tensor X, boolean training) {
        throw new UnsupportedOperationException("DecoderBlock requires (X, encoderOut, selfMask, crossMask)");
    }

    @Override
    public Tensor forwardMany(boolean training, Tensor... xs) {
        if(xs == null || xs.length != 4)
            throw new IllegalArgumentException();

        Tensor tgtX = xs[0]; Tensor srcX = xs[1]; Tensor tgtMask = xs[2]; Tensor srcMask = xs[3];
        if(tgtMask == null || srcMask == null)
            throw new IllegalArgumentException("mask null");

        // Self-Attention
        Tensor N1 = norm1.forward(tgtX, training);
        Tensor selfAttn = selfMHA.forwardMany(training, N1, N1, tgtMask);
        // Residual connection
        Tensor Y = tgtX.add(selfAttn);

        // Cross-Attention
        Tensor N2 = norm2.forward(Y, training);
        Tensor ca = crossMHA.forwardMany(training, N2, srcX, srcMask);
        // Residual connection
        Tensor Z = Y.add(ca);

        // FFN
        Tensor N3 = norm3.forward(Z, training);
        Tensor f = ffn.forward(N3, training);
        // Residual connection
        Tensor O = Z.add(f);

        return O;
    }

    @Override
    public Tensor calcGradients(Tensor dY, boolean accumulate, double scale) {
        throw new UnsupportedOperationException("Use calcGradientsMany()");
    }

    @Override
    public Tensor[] calcGradientsMany(Tensor dY, boolean accumulate, double scale) {
        Tensor dY_s = (scale == 1.0) ? dY : dY.mul(scale);

        // O = Z + f
        Tensor dZ = ffn.calcGradients(dY_s, accumulate, 1.0);
        Tensor dNorm3 = norm3.calcGradients(dZ, accumulate, 1.0);
        Tensor dR1 = dZ.add(dNorm3);

        // Z = Y + ca
        Tensor[] gradCross = crossMHA.calcGradientsMany(dR1, accumulate, 1.0);
        Tensor dNorm2 = norm2.calcGradients(gradCross[0], accumulate, 1.0);
        Tensor dEncOut = gradCross[1];
        Tensor dR2 = dR1.add(dNorm2);

        // Y = X + sa
        Tensor[] gradSelf = selfMHA.calcGradientsMany(dR2, accumulate, 1.0);
        Tensor dX1 = gradSelf[0].add(gradSelf[1]);
        Tensor dNorm1 = norm1.calcGradients(dX1, accumulate, 1.0);
        Tensor dX = dR2.add(dNorm1);

        return new Tensor[]{dX, dEncOut};
    }

    @Override
    public void update(Optimizer optimizer) {
        if(isTrainable) {
            norm1.update(optimizer); norm2.update(optimizer); norm3.update(optimizer);
            selfMHA.update(optimizer); crossMHA.update(optimizer);
            ffn.update(optimizer);
        }
    }

    @Override
    public List<Parameter> parameters() {
        List<Parameter> ps = new ArrayList<>();
        ps.addAll(norm1.parameters()); ps.addAll(norm2.parameters()); ps.addAll(norm3.parameters());
        ps.addAll(selfMHA.parameters()); ps.addAll(crossMHA.parameters());
        ps.addAll(ffn.parameters());

        return ps;
    }

    @Override
    public void zeroGrad() {
        norm1.zeroGrad(); norm2.zeroGrad(); norm3.zeroGrad();
        selfMHA.zeroGrad(); crossMHA.zeroGrad();
        ffn.zeroGrad();
    }
}
