package nn.activation;

import tensor.Tensor;

public class Tanh implements Activation {

    private Tensor cY;

    @Override
    public Tensor forward(Tensor X) {
        this.cY = X.tanh();
        return cY;
    }

    @Override
    public Tensor backward(Tensor dY) {
        return dY.mul(cY.pow(2.0).neg().add(1.0));
    }
}
