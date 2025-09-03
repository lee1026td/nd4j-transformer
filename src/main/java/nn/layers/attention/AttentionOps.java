package nn.layers.attention;

import tensor.Tensor;

public class AttentionOps {

    public static Tensor qkScores(Tensor Q, Tensor K, double scale) {
        return Q.matmul(K.transpose(-1, -2)).div(scale);
    }

    // Additive mask
    public static Tensor applyMask(Tensor attnScores, Tensor mask) {
        return attnScores.add(mask);
    }

    public static Tensor softmaxLast(Tensor attnScores) {
        Tensor max = attnScores.max(-1, true);
        Tensor shifted = attnScores.sub(max).exp();
        Tensor denom = shifted.sum(-1, true).add(1e-12);

        return shifted.div(denom);
    }

    public static Tensor splitHeads(Tensor X, int H) {

        // X : [B, T, d_model], where d_model = h * d_h
        int B = X.size(0);
        int T = X.size(-2);

        if(X.size(-1) % H != 0)
            throw new IllegalArgumentException("Last dimension is not divisible by H");

        int d_h = X.size(-1) / H;

        // [B, T, h, d_h]
        Tensor tmp = X.reshape(B, T, H, d_h);

        // [B, h, T, d_h]
        return tmp.transpose(-3, -2);
    }

    public static Tensor mergeHeads(Tensor O) {
        // O : [B, h, T, d_h] -> tmp : [B, T, h, d_h]
        Tensor tmp = O.transpose(-3, -2);

        int B = tmp.size(0);
        int T = tmp.size(-3);
        int h = tmp.size(-2);
        int d_h = tmp.size(-1);

        // Merge : [B, T, h * d_h], where h * d_h = d_model
        return tmp.reshape(B, T, h * d_h);
    }
}
