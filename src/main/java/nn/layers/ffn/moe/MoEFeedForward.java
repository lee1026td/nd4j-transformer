package nn.layers.ffn.moe;

import nn.core.Parameter;
import nn.core.activation.Activation;
import nn.core.initializer.Initializer;
import nn.core.optimizer.Optimizer;
import nn.layers.ffn.FeedForward;
import nn.layers.ffn.FeedForwardNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import tensor.Tensor;

import java.util.List;

import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

public class MoEFeedForward implements FeedForward {

    private final int d_model, d_hidden;
    private final int numExperts, topK;
    private final boolean useNoisyTopK;
    private final double capacityFactor;

    private final FeedForwardNetwork[] experts;
    private final MoERouter router;

    private int expCapacity;
    private int B, T;

    public MoEFeedForward(int d_model, int d_hidden, int numExperts, int topK, boolean useNoisyTopK, double capacityFactor, Activation act, Initializer wInit, Initializer bInit, boolean isTrainable, double dropProb) {
        this.d_model = d_model;
        this.d_hidden = d_hidden;
        this.numExperts = numExperts;
        this.topK = topK;
        this.useNoisyTopK = useNoisyTopK;
        this.capacityFactor = capacityFactor;

        this.experts = new FeedForwardNetwork[numExperts];
        for(int i=0;i<numExperts;i++) {
            experts[i] = new FeedForwardNetwork(d_model, d_hidden, act, wInit, bInit, isTrainable, dropProb);
        }

        this.router = new MoERouter(d_model, numExperts, topK, useNoisyTopK, capacityFactor, wInit, isTrainable);
    }

    @Override
    public Tensor forward(Tensor X, boolean training) {
        router.forward(X, training);

        // Get dispatch, combine, expCapacity
        Tensor dispatch = router.getDispatch();
        Tensor combine = router.getCombine();
        this.expCapacity = router.getExpCapacity();

        // Expert input batch
        // dispatch [B*T, N, C] -> transpose(0, -1) [C, N, B*T] -> to 2D [C*N, B*T]
        // X [B*T, d]
        // dispatch^T (2D) x X => [C*N, d]
        // -> to original shape : [N, C, D]
        this.B = X.size(0);
        this.T = X.size(1);
        int numTokens = B * T;
        Tensor dispatchT2D = dispatch.permute(1, 2, 0).reshape(numExperts*expCapacity, numTokens);
        Tensor out2D = dispatchT2D.matmul(X);
        Tensor expertIn = out2D.reshape(numExperts, expCapacity, d_model);

        // Run Experts
        INDArray expYArr = Nd4j.createUninitialized(expertIn.getNDArray().dataType(), numExperts, expCapacity, d_model);
        for(int i=0;i<numExperts;i++) {
            Tensor Xe = new Tensor(expertIn.getNDArray().get(point(i), all(), all()));
            Tensor Ye = experts[i].forward(Xe, training);
            expYArr.put(new INDArrayIndex[]{point(i), all(), all()}, Ye.getNDArray());
        }

        // Combine

    }

    @Override
    public Tensor calcGradients(Tensor dY, boolean accumulate, double scale) {
        return null;
    }

    @Override
    public void update(Optimizer optimizer) {

    }

    @Override
    public List<Parameter> parameters() {
        return List.of();
    }

    @Override
    public void zeroGrad() {

    }
}
