package nn.transformer;

import nn.core.Module;
import nn.core.Parameter;
import nn.layers.Linear;
import nn.optimizer.Optimizer;
import nn.transformer.embeddings.SinusoidalPositionalEncoding;
import nn.transformer.embeddings.TokenEmbeddings;
import nn.transformer.mask.MaskUtils;
import nn.transformer.modules.Decoder;
import nn.transformer.modules.Encoder;
import tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

public class Transformer implements Module {

    private final TokenEmbeddings srcEmb, tgtEmb;
    private final SinusoidalPositionalEncoding posEnc;
    private final Encoder encoder;
    private final Decoder decoder;
    private final Linear lmHead;

    private final int padId, bosId, eosId;

    public Transformer(TokenEmbeddings srcEmb,
                       TokenEmbeddings tgtEmb,
                       SinusoidalPositionalEncoding posEnc,
                       Encoder encoder, Decoder decoder,
                       Linear lmHead,
                       int padId, int bosId, int eosId) {
        this.srcEmb = srcEmb; this.tgtEmb = tgtEmb;
        this.posEnc = posEnc;
        this.encoder = encoder; this.decoder = decoder;
        this.lmHead = lmHead;

        this.padId = padId; this.bosId = bosId; this.eosId = eosId;
    }

    @Override
    public Tensor forward(Tensor X, boolean training) {
        return null;
    }

    @Override
    public Tensor forwardMany(boolean training, Tensor... xs) {
        return forward(xs[0], xs[1], training);
    }

    @Override
    public Tensor calcGradients(Tensor dLogits, boolean accumulate, double scale) {
        backward(dLogits, accumulate, scale);

        return dLogits;
    }


    // Forward for training (Teacher forcing)
    public Tensor forward(Tensor srcIds, Tensor tgtIds, boolean training) {
        int[] srcLens = lengthsFromIds(srcIds, padId);
        int[] tgtLens = lengthsFromIds(tgtIds, padId);

        // Embeddings + positional
        Tensor srcX = posEnc.apply(srcEmb.forward(srcIds, training));
        Tensor tgtX = posEnc.apply(tgtEmb.forward(tgtIds, training));

        // Masks
        int B = srcIds.size(0);
        int S = srcIds.size(1);
        int T = tgtIds.size(1);

        Tensor encMask = MaskUtils.padMaskFromLength(srcLens, S, S);        // [B, 1, S, S]
        Tensor causal = MaskUtils.causalMask(B, T);                         // [B, 1, T, T]
        Tensor tgtPadMask = MaskUtils.padMaskFromLength(tgtLens, T, T);     // [B, 1, T, T]
        Tensor decSelfMask = causal.add(tgtPadMask);                        // [B, 1, T, T]
        Tensor crossMask = MaskUtils.padMaskFromLength(srcLens, T, S);      // [B, 1, T, S]

        // Encoder / Decoder
        Tensor encOut = encoder.forwardMany(training, srcX, encMask);
        Tensor Y = decoder.forwardMany(training, tgtX, encOut, decSelfMask, crossMask);

        // Projection
        Tensor logits = lmHead.forward(Y, training);
        return logits;
    }

    public void backward(Tensor dLogits, boolean accumulate, double scale) {
        Tensor dY = lmHead.calcGradients(dLogits, accumulate, scale);

        // Decoder
        Tensor[] gDec = decoder.calcGradientsMany(dY, accumulate, 1.0);
        Tensor dTgtX = gDec[0];
        Tensor dEncOut = gDec[1];

        // Encoder
        Tensor dSrcX = encoder.calcGradients(dEncOut, accumulate, 1.0);

        // Embeddings
        tgtEmb.calcGradients(dTgtX, accumulate, 1.0);
        srcEmb.calcGradients(dSrcX, accumulate, 1.0);
    }


    @Override
    public void update(Optimizer optimizer) {
        srcEmb.update(optimizer);
        tgtEmb.update(optimizer);
        encoder.update(optimizer);
        decoder.update(optimizer);
        lmHead.update(optimizer);
    }

    @Override
    public List<Parameter> parameters() {
        List<Parameter> ps = new ArrayList<>();
        ps.addAll(srcEmb.parameters()); ps.addAll(tgtEmb.parameters());
        ps.addAll(encoder.parameters()); ps.addAll(decoder.parameters());
        ps.addAll(lmHead.parameters());

        return ps;
    }

    @Override
    public void zeroGrad() {
        srcEmb.zeroGrad();
        tgtEmb.zeroGrad();
        encoder.zeroGrad();
        decoder.zeroGrad();
        lmHead.zeroGrad();
    }

    private int[] lengthsFromIds(Tensor ids, int padId) {
        org.nd4j.linalg.api.ndarray.INDArray a = ids.getNDArray(); // [B,T]
        int B = (int) a.size(0), T = (int) a.size(1);
        int[] lens = new int[B];
        for (int b = 0; b < B; b++) {
            int len = T;
            // scan from right to left until first non-PAD
            for (int t = T - 1; t >= 0; t--) {
                int v = (int) a.getDouble(b, t);
                if (v != padId) { len = t + 1; break; }
                if (t == 0) len = 0;
            }
            lens[b] = len;
        }
        return lens;
    }
}
