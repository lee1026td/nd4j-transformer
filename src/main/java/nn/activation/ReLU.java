package nn.activation;

import tensor.Tensor;

public class ReLU implements Activation {

    private Tensor cX;

    @Override
    public Tensor forward(Tensor X) {
        this.cX = X;
        return X.relu();
    }

    @Override
    public Tensor backward(Tensor dY) {
        return dY.mul(cX.gt(0.0));
    }
}
