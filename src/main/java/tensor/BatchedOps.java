package tensor;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

public class BatchedOps {

    // [B, M, K] x [K, N] = [B, M, N]
    public static Tensor mmulND2(Tensor X, Tensor W) {
        int r = X.ndim();
        if(r < 3) throw new IllegalArgumentException("X.ndim must be >= 3");

        int M = X.size(r - 2);
        int K = X.size(r - 1);

        if(W.ndim() != 2 || W.size(0) != K)
            throw new IllegalArgumentException("W.ndim must be == 2 with [K, N], K == X.lastDim");

        int N = W.size(1);

        // Batch
        int B = 1;
        for(int i=0;i<r-2;i++) B *= X.size(i);

        // Reshape : [B, M, K] -> [B*M, K]
        Tensor X2D = X.reshape(B * M, K);

        // GEMM 사용
        INDArray X2DA = X2D.getNDArray();
        INDArray WA = W.getNDArray();
        INDArray Z2DA = Nd4j.createUninitialized(new long[]{(long)B * M, N}, 'c');
        Nd4j.gemm(X2DA, WA, Z2DA, false, false, 1.0, 0.0);

        int[] out = X.shape().clone();
        out[r - 1] = N;

        return new Tensor(Z2DA.reshape(out));
    }

    // [..., M, K] x [..., K, N] = [..., M, N]
    public static Tensor bmmulND2(Tensor X1, Tensor X2) {
        if(X1.ndim() != X2.ndim() || X1.ndim() < 3)
            throw new IllegalArgumentException("Batched Matmul expects X1, X2 to have the same rank >= 3");

        for(int i=0;i<X1.ndim() - 2;i++) {
            if(X1.size(i) != X2.size(i))
                throw new IllegalArgumentException("Shape mismatch : " + "X1 : " + X1.size(i) + ", X2 : " + X2.size(i));
        }

        final int M = X1.size(-2);
        final int K = X1.size(-1);
        final int K2 = X2.size(-2);
        final int N = X2.size(-1);

        if(K != K2) throw new IllegalArgumentException("Shape mismatch");

        long Bstar = 1L;
        for(int i=0;i<X1.ndim()-2;i++) Bstar *= X1.size(i);

        // [B*, M, K], [B*, K, N]
        Tensor _X1 = X1.reshape(Math.toIntExact(Bstar), M, K);
        Tensor _X2 = X2.reshape(Math.toIntExact(Bstar), K, N);

        // Out
        INDArray Y = Nd4j.createUninitialized(new long[]{Bstar, M, N}, 'c');

        INDArray _X1A = _X1.getNDArray();
        INDArray _X2A = _X2.getNDArray();
        for(int b=0;b<Bstar;b++) {
            // [M, K] x [K, N] = [M, N]
            INDArray X1b = _X1A.slice(b);
            INDArray X2b = _X2A.slice(b);
            INDArray Yb = Y.slice(b);

            // Yb = X1b * X2b
            Nd4j.gemm(X1b, X2b, Yb, false, false, 1.0, 0.0);

        }

        int[] out = X1.shape().clone();
        out[X1.ndim() - 2] = M;
        out[X1.ndim() - 1] = N;

        return new Tensor(Y.reshape('c', out));
    }
}
