package nn.activation;

import tensor.Tensor;

public class Sigmoid implements Activation {

    private Tensor cY;

    @Override
    public Tensor forward(Tensor X) {
        this.cY = X.sigmoid();
        return cY;
    }

    @Override
    public Tensor backward(Tensor dY) {
        return dY.mul(cY.mul(cY.neg().add(1.0)));
    }
}
