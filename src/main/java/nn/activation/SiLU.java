package nn.activation;

import tensor.Tensor;

public class SiLU implements Activation {

    private Tensor cSigmoid;
    private Tensor cY;

    @Override
    public Tensor forward(Tensor X) {
        this.cSigmoid = X.sigmoid();
        this.cY = X.mul(cSigmoid);
        return cY;
    }

    @Override
    public Tensor backward(Tensor dY) {
        Tensor grad = cSigmoid.add(cSigmoid.neg().add(1.0).mul(cY));
        return dY.mul(grad);
    }
}
