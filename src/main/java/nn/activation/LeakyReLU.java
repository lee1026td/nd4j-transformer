package nn.activation;

import tensor.Tensor;

public class LeakyReLU implements Activation {

    private final double alpha;
    private Tensor cX;

    public LeakyReLU(double alpha) {
        this.alpha = alpha;
    }

    @Override
    public Tensor forward(Tensor X) {
        this.cX = X;
        return X.leakyRelu(alpha);
    }

    @Override
    public Tensor backward(Tensor dY) {
        return dY.mul(cX.gt(0.0).add(cX.le(0.0).mul(alpha)));
    }
}
