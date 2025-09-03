package nn.layers;

import nn.core.Module;
import nn.core.Parameter;
import nn.core.initializer.Initializer;
import nn.core.optimizer.Optimizer;
import nn.mask.MaskUtils;
import tensor.Tensor;

import java.util.List;
import java.util.stream.IntStream;

public class Linear implements Module {

    private final int inFeatures, outFeatures;
    private Parameter W;
    private Parameter b;
    private Tensor X;
    private Tensor cDrop;

    private final boolean isTrainable;
    private final double dropoutProb;

    private boolean useBias;

    public Linear(int inFeatures, int outFeatures, Initializer wInit, Initializer bInit, boolean isTrainable, double dropoutProb) {
        this.inFeatures = inFeatures;
        this.outFeatures = outFeatures;
        this.isTrainable = isTrainable;
        this.dropoutProb = dropoutProb;
        this.W = new Parameter("linear.W", wInit.init(inFeatures, outFeatures), isTrainable);

        if(bInit != null) {
            this.b = new Parameter("linear.b", bInit.init(1, outFeatures), isTrainable);
            this.useBias = true;
        } else this.useBias = false;
    }

    public Linear(int inFeatures, int outFeatures, Initializer wInit, Initializer bInit, double dropoutProb) {
        this(inFeatures, outFeatures, wInit, bInit, true, dropoutProb);
    }

    public Linear(int inFeatures, int outFeatures, Initializer wInit, Initializer bInit, boolean isTrainable) {
        this(inFeatures, outFeatures, wInit, bInit, isTrainable, 0.0);
    }

    public Linear(int inFeatures, int outFeatures, Initializer wInit, boolean isTrainable) {
        this(inFeatures, outFeatures, wInit, null, isTrainable, 0.0);
    }

    public Linear(int inFeatures, int outFeatures, Initializer wInit, boolean isTrainable, double dropoutProb) {
        this(inFeatures, outFeatures, wInit, null, isTrainable, dropoutProb);
    }

    @Override
    public Tensor forward(Tensor X, boolean training) {
        this.X = X;

        // Y = XW
        Tensor Y = X.matmul(W.getData());

        // if b is not null : Y = XW + b
        if(useBias) {
            Y = Y.add(b.getData());
        }

        if(training) {
            cDrop = MaskUtils.dropoutMaskLike(Y, dropoutProb);
            Y = Y.mul(cDrop);
        }
        else cDrop = null;

        return Y;
    }

    @Override
    public Tensor calcGradients(Tensor dY, boolean accumulate, double scale) {
        if(scale != 1.0) dY = dY.mul(scale);
        if(cDrop != null) dY = dY.mul(cDrop);

        // dL/dW = X^T x dY
        Tensor dW = X.transpose(-1, -2).matmul(dY);
        // Considering batch dimension
        int[] axesDW = IntStream.range(0, dW.ndim() - 2).toArray();
        dW = (axesDW.length > 0) ? dW.sum(false, axesDW) : dW;

        // dL/db = sum(dY), Along all axis w/o last axis : [1, 1, d_model]
        int[] axesDB = IntStream.range(0, dY.ndim() -1).toArray();
        Tensor db = dY.sum(true, axesDB);
        // dL/dX = dY x W^T
        Tensor dX = dY.matmul(W.getData().transpose(-1, -2));

        if(accumulate) {
            // backwardAccumulate()
            W.addGrad(dW);
            if(useBias) b.addGrad(db);
        } else {
            // backward()
            W.setGrad(dW);
            if(useBias) b.setGrad(db);
        }

        return dX;
    }

    @Override
    public void update(Optimizer optimizer) {
        if(isTrainable) {
            optimizer.step(W);
            if(useBias) optimizer.step(b);
        }
    }

    @Override
    public List<Parameter> parameters() {
        if(useBias) {
            return List.of(W, b);
        } else return List.of(W);
    }

    @Override
    public void zeroGrad() {
        W.zeroGrad();
        if(useBias) b.zeroGrad();
        X = null;
    }

    public void setParameter(Parameter parameter) {
        this.W = parameter;
    }
}
