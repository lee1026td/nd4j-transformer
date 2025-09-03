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

public class Decoder implements Module {

    private final DecoderBlock[] decoderBlocks;
    private final boolean isTrainable;

    private Tensor encOut;  // Last EncoderBlock's output

    public Decoder(int d_model,
                   int d_hidden,
                   int numHeads,
                   int numDecoders,
                   Supplier<? extends Module> norm,
                   boolean isTrainable,
                   double mhaDropProb,
                   double ffnDropProb) {
        this.decoderBlocks = new DecoderBlock[numDecoders];
        this.isTrainable = isTrainable;

        for(int i=0;i<numDecoders;i++) {
            decoderBlocks[i] = new DecoderBlock(
                    norm.get(),
                    norm.get(),
                    norm.get(),
                    new MultiHeadAttention(d_model, numHeads, new XavierNormal(), isTrainable, mhaDropProb, mhaDropProb, false),
                    new MultiHeadAttention(d_model, numHeads, new XavierNormal(), isTrainable, mhaDropProb, mhaDropProb, true),
                    new FeedForwardNetwork(d_model, d_hidden, new GELU(true), new XavierNormal(), new HeNormal(), isTrainable, ffnDropProb),
                    isTrainable
            );
        }
    }

    @Override
    public Tensor forward(Tensor X, boolean training) {
        throw new UnsupportedOperationException("Decoder expects (X, encOut, tgtMask, srcMask)");
    }

    @Override
    public Tensor forwardMany(boolean training, Tensor... xs) {
        if(xs == null || xs.length != 4)
            throw new IllegalArgumentException();
        Tensor out = xs[0]; Tensor encOut = xs[1]; Tensor selfMask = xs[2]; Tensor crossMask = xs[3];

        this.encOut = encOut;
        if(selfMask == null || crossMask == null)
            throw new IllegalArgumentException("masks null");

        for(DecoderBlock block : decoderBlocks) {
            out = block.forwardMany(training, out, encOut, selfMask, crossMask);
        }

        return out;
    }

    @Override
    public Tensor calcGradients(Tensor dY, boolean accumulate, double scale) {
        throw new UnsupportedOperationException("Use calcGradientsMany()");
    }

    @Override
    public Tensor[] calcGradientsMany(Tensor dY, boolean accumulate, double scale) {
        Tensor dY_s = (scale == 1.0) ? dY : dY.mul(scale);

        Tensor dEncOutSum = Tensor.zeros(encOut.shape());
        for(int i=decoderBlocks.length - 1;i>=0;i--) {
            Tensor[] grads = decoderBlocks[i].calcGradientsMany(dY_s, accumulate, scale);

            // Gradients from next DecoderBlock
            dY_s = grads[0];

            // Sum all the gradients for DecoderBlocks, passed to last hidden Encoderblock
            dEncOutSum = dEncOutSum.add(grads[1]);
        }

        return new Tensor[]{dY_s, dEncOutSum};
    }

    @Override
    public void update(Optimizer optimizer) {
        if(isTrainable) for(DecoderBlock block : decoderBlocks) block.update(optimizer);
    }

    @Override
    public List<Parameter> parameters() {
        List<Parameter> ps = new ArrayList<>();
        for(DecoderBlock block : decoderBlocks) ps.addAll(block.parameters());

        return ps;
    }

    @Override
    public void zeroGrad() {
        for(DecoderBlock block : decoderBlocks) block.zeroGrad();
    }
}
