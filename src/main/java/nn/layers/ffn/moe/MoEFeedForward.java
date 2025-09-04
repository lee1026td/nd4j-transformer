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

    private final int d_model;
    private final int numExperts;
    private final int topK;

    private final FeedForwardNetwork[] experts;
    private final MoERouter router;

    private int expCapacity;
    private int B, T;

    private Tensor dispatchT2D, combine2D;

    public MoEFeedForward(int d_model, int d_hidden, int numExperts, int topK, boolean useNoisyTopK, double capacityFactor, Activation act, Initializer wInit, Initializer bInit, boolean isTrainable, double dropProb) {
        this.d_model = d_model;
        this.numExperts = numExperts;
        this.topK = topK;

        this.experts = new FeedForwardNetwork[numExperts];
        for(int i=0;i<numExperts;i++) {
            experts[i] = new FeedForwardNetwork(d_model, d_hidden, act, wInit, bInit, isTrainable, dropProb);
        }

        this.router = new MoERouter(d_model, numExperts, topK, useNoisyTopK, capacityFactor, wInit, isTrainable);
    }

    @Override
    public Tensor forward(Tensor X, boolean training) {
        // Runs MoERouter first to routes each input tokens to experts
        router.forward(X, training);

        // Get dispatch, combine, expCapacity from MoERouter cache
        Tensor dispatch = router.getDispatch();
        Tensor combine = router.getCombine();
        this.expCapacity = router.getExpCapacity();

        // Expert input batch
        this.B = X.size(0);
        this.T = X.size(1);
        int numTokens = B * T;

        // dispatch [B*T, N, C] -> 2D [B*T, N*C] -> transpose [N*C, B*T]
        this.dispatchT2D = dispatch.reshape(numTokens, numExperts*expCapacity).transpose();
        // X [B, T, d] -> 2D [B*T, d]
        Tensor X2D = X.reshape(numTokens, d_model);
        // dispatch^T (2D) x X => [N*C, d]
        // Making each expert's input batch
        Tensor out2D = dispatchT2D.matmul(X2D);
        // out2D [N*C, d] -> expertIn [N, C, d]
        Tensor expertIn = out2D.reshape(numExperts, expCapacity, d_model);

        // Run Experts
        INDArray expYArr = Nd4j.createUninitialized(expertIn.getNDArray().dataType(), numExperts, expCapacity, d_model);
        for(int i=0;i<numExperts;i++) {
            Tensor Xe = new Tensor(expertIn.getNDArray().get(point(i), all(), all()));
            Tensor Ye = experts[i].forward(Xe, training);
            expYArr.put(new INDArrayIndex[]{point(i), all(), all()}, Ye.getNDArray());
        }

        // [N, C, d]
        Tensor expY = new Tensor(expYArr);
        System.out.println("expY : \n" + expY);

        // Combine
        // Gathers all experts outputs to one single output

        // combine [B*T, N, C] -> 2D [B*T, N*C]
        this.combine2D = combine.reshape(numTokens, numExperts*expCapacity);  // [B*T, N*C]
        // expY [N, C, d] -> 2D [N*C, d]
        Tensor expY2D = expY.reshape(numExperts*expCapacity, d_model);          // [N*C, d]
        // Each output of expert's slot -> to original minibatch shape projection
        // combine2D x expY2D = [B*T, N*C] x [N*C, d] = [B*T, d]
        Tensor Y2D = combine2D.matmul(expY2D);
        Tensor Y = Y2D.reshape(B, T, d_model);  // [B, T, d]

        return Y;
    }

    @Override
    public Tensor calcGradients(Tensor dY, boolean accumulate, double scale) {
        // dY -> to each experts
        int numTokens = B * T;
        // Reshape dY [B, T, d] -> [B*T, d]
        Tensor dY2D = dY.reshape(numTokens, d_model);

        // 1. (cached) combine2D : [B*T, N*C] -> transpose [N*C, B*T]
        // 2. combine2D^T x dY = [N*C, B*T] x [B*T, d] = [N*C, d]
        Tensor dYpacked2D = combine2D.transpose().matmul(dY2D);
        // 3. Reshape to 3D : [N, C, d]
        Tensor dYpacked = dYpacked2D.reshape(numExperts, expCapacity, d_model);

        // Expert backward
        // dXpacked : [N, C, d]
        Tensor dXpacked = Tensor.zeros(numExperts, expCapacity, d_model);
        for(int i=0;i<numExperts;i++) {
            // to each expert : get row [C, d]
            Tensor dY_e = new Tensor(dXpacked.getNDArray().get(point(i), all(), all()));

            Tensor dX_e = experts[i].calcGradients(dY_e, accumulate, scale);

            // Assign to packed buffer
            dXpacked.getNDArray().get(point(i), all(), all()).assign(dX_e.getNDArray());
        }

        // 1. dXpacked [N, C, d] -> 2D [N*C, d]
        Tensor dXpacked2D = dXpacked.reshape(numExperts*expCapacity, d_model);
        // 2. dX2D = dispatchT2D x dXpacked2D = [B*T, N*C] x [N*C, d] = [B*T, d]
        Tensor dX2D = dispatchT2D.matmul(dXpacked2D);
        // 3. Reshape [B*T, d] -> [B, T, d]
        Tensor dX = dX2D.reshape(B, T, d_model);

        return dX;
    }

    @Override
    public void update(Optimizer optimizer) {
        for(FeedForwardNetwork expert : experts) {
            expert.update(optimizer);
        }
    }

    @Override
    public List<Parameter> parameters() {
        return List.of();
    }

    @Override
    public void zeroGrad() {
        for(FeedForwardNetwork expert : experts) {
            expert.zeroGrad();
        }
    }

    public MoERouter getRouter() {
        return router;
    }
}
