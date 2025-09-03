package nn.mask;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import tensor.Tensor;

import static org.nd4j.linalg.indexing.NDArrayIndex.*;

public class MaskUtils {

    public static final double NEG_INF = -1e4;

    // Causal mask : [1, 1, T, T] -> Batch broadcastable
    public static Tensor causalMask(int T) {
        INDArray m = Nd4j.valueArrayOf(new long[]{1, 1, T, T}, 0.0);
        for(int i=0;i<T;i++) {
            for(int j=i+1;j<T;j++) {
                m.putScalar(new int[]{0, i, j}, NEG_INF);
            }
        }

        return new Tensor(m);
    }

    // Causal mask : [B, 1, T, T]
    public static Tensor causalMask(int B, int T) {
        INDArray m = Nd4j.valueArrayOf(new long[]{B, 1, T, T}, 0.0);
        for(int b=0;b<B;b++) {
            for(int i=0;i<T;i++) {
                for(int j=i+1;j<T;j++) {
                    m.putScalar(new int[]{b, 0, i, j}, NEG_INF);
                }
            }
        }

        return new Tensor(m);
    }

    // Causal mask (generalized) : [B, 1, Tq, Tk] (for decoder's self-attention)
    public static Tensor causalMask(int B, int Tq, int Tk) {
        INDArray m = Nd4j.valueArrayOf(new long[]{B, 1, Tq, Tk}, 0.0);
        for(int b=0;b<B;b++) {
            for(int i=0;i<Tq;i++) {
                for(int j=0;j<Tk;j++) {
                    if(j > i) m.putScalar(new int[]{b, 0, i, j}, NEG_INF);;
                }
            }
        }
        return new Tensor(m);
    }

    /**
     * Pad mask from key length : [B, Tq, Tk]
     * @param keyLens is valid lengths per each sequence on batch.
     * @param Tq is max length for Query input, including "pad" tokens
     * @param Tk is max length for Key/Value input, including "pad" tokens
     *
     */
    public static Tensor padMaskFromLength(int[] keyLens, int Tq, int Tk) {
        int B = keyLens.length;
        INDArray m = Nd4j.valueArrayOf(new long[]{B, 1, Tq, Tk}, 0.0);
        for(int b=0;b<B;b++) {
            int L = keyLens[b];
            if(L < 0 || L > Tk)
                throw new IllegalArgumentException("keyLens["+b+"] out of range : " + L + " vs Tk = " + Tk);

            if(L < Tk) {
                m.get(point(b), all(), all(), interval(L, Tk)).assign(NEG_INF);
            }
        }
        return new Tensor(m);
    }

    // Dropout mask (inverted) : keep = 1-p, keep -> 1/(1-p), drop -> 0
    public static Tensor dropoutMaskLike(Tensor X, double dropProb) {
        if(dropProb <= 0.0) return Tensor.ones(X.shape());
        if(dropProb >= 1.0) return Tensor.zeros(X.shape());

        double keep = 1.0 - dropProb;

        return Tensor.randomBernoulli(keep, X.shape()).div(keep);
    }

    // For use in Loss calculation, except PAD tokens
    public static Tensor lengthsToMask(int[] lens, int T){
        int B = lens.length;
        INDArray m = Nd4j.zeros(B, T);
        for (int b=0; b<B; b++) {
            int n = Math.min(lens[b], T);
            if (n > 0) m.getRow(b).get(NDArrayIndex.interval(0, n)).assign(1.0);
        }
        return new Tensor(m);
    }
}
