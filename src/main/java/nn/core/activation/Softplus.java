package nn.core.activation;

import tensor.Tensor;

public class Softplus implements Activation {

    private Tensor cX;

   @Override
    public Tensor forward(Tensor X) {
        this.cX = X;
        return X.exp().add(1.0).log();
    }

    @Override
    public Tensor backward(Tensor dY) {
        return new Sigmoid().forward(cX).mul(dY);
    }
}
