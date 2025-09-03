package nn.modules;

import nn.core.activation.GELU;
import nn.core.Module;
import nn.core.Parameter;
import nn.core.initializer.HeNormal;
import nn.core.initializer.XavierNormal;
import nn.core.optimizer.Optimizer;
import nn.layers.attention.MultiHeadAttention;
import nn.layers.ffn.FeedForwardNetwork;
import tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

public class Encoder implements Module {

    private final EncoderBlock[] encoderBlocks;
    private final boolean isTrainable;

    public Encoder(int d_model,
                   int d_hidden,
                   int numHeads,
                   int numEncoders,
                   Supplier<? extends Module> norm,
                   boolean isTrainable,
                   double mhaDropProb,
                   double ffnActDropProb) {
        this.encoderBlocks = new EncoderBlock[numEncoders];
        this.isTrainable = isTrainable;

        for(int i=0;i<numEncoders;i++) {
            encoderBlocks[i] = new EncoderBlock(
                    norm.get(),
                    norm.get(),
                    new MultiHeadAttention(d_model, numHeads, new XavierNormal(), isTrainable, mhaDropProb, mhaDropProb, false),
                    new FeedForwardNetwork(d_model, d_hidden, new GELU(true), new XavierNormal(), new HeNormal(), isTrainable, ffnActDropProb),
                    isTrainable
            );
        }
    }

    @Override
    public Tensor forward(Tensor X, boolean training) {
        throw new UnsupportedOperationException("Encoder expects (X, mask)");
    }

    @Override
    public Tensor forwardMany(boolean training, Tensor... xs) {
        Tensor out = xs[0]; Tensor mask = xs[1];

        if(mask == null) throw new IllegalArgumentException("mask null");

        for(EncoderBlock block : encoderBlocks) {
            out = block.forwardMany(training, out, mask);
        }

        return out;
    }

    @Override
    public Tensor calcGradients(Tensor dY, boolean accumulate, double scale) {
        Tensor dY_s = (scale == 1.0) ? dY : dY.mul(scale);
        Tensor g = dY_s;
        for(int i=encoderBlocks.length - 1;i>=0;i--) {
            g = encoderBlocks[i].calcGradients(g, accumulate, 1.0);
        }

        return g;
    }

    @Override
    public void update(Optimizer optimizer) {
        if(isTrainable) for(EncoderBlock block : encoderBlocks) block.update(optimizer);
    }

    @Override
    public List<Parameter> parameters() {
        List<Parameter> ps = new ArrayList<>();
        for(EncoderBlock block : encoderBlocks) ps.addAll(block.parameters());

        return ps;
    }

    @Override
    public void zeroGrad() {
        for(EncoderBlock block : encoderBlocks) block.zeroGrad();
    }
}
