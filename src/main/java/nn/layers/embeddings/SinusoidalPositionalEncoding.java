package nn.layers.embeddings;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import tensor.Tensor;

import static org.nd4j.linalg.indexing.NDArrayIndex.*;

public class SinusoidalPositionalEncoding {

    private final int maxLength, d_model;
    private final Tensor table;       // [maxLen, d]

    public SinusoidalPositionalEncoding(int maxLength, int d_model) {
        this.maxLength = maxLength;
        this.d_model = d_model;

        this.table = buildTable(maxLength, d_model);
    }

    private static Tensor buildTable(int L, int d) {
        INDArray pe = Nd4j.create(new long[]{L, d});

        // Even index of embedded vector : pe[pos, 2i] = sin(pos / 10000^{2i / d})
        // Odd index of embedded vector : pe[pos, 2i+1] = cos(pos / 10000^{2i / d})
        for(int pos=0;pos<L;pos++) {
            for(int i=0;i<d;i+=2) {
                double ang = pos / Math.pow(10000.0, (double) i / d);
                pe.putScalar(pos, i, Math.sin(ang));

                if(i + 1 < d) pe.putScalar(pos, i + 1, Math.cos(ang));
            }
        }

        return new Tensor(pe);
    }

    // X : [B, T, d] -> X + PE[:T]
    public Tensor apply(Tensor X) {
        int T = X.size(1);

        if(T > maxLength) throw new IllegalArgumentException("T("+T+") > maxLength("+maxLength+")");

        INDArray pos = table.getNDArray().get(interval(0, T), all());           // [T, d]
        INDArray bCastPos = pos.reshape('c', 1, T, d_model);    // [1, T, d]

        return X.add(new Tensor(bCastPos));
    }

}
