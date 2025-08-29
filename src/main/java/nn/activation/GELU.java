package nn.activation;

import tensor.Tensor;

public class GELU implements Activation {

    private final boolean precise;
    private Tensor cX;

    public GELU(boolean precise) {
        this.precise = precise;
    }

    @Override
    public Tensor forward(Tensor X) {
        this.cX = X;
        return precise ? X.geluExact() : X.geluApprox();
    }

    @Override
    public Tensor backward(Tensor dY) {
        return dY.mul(precise ? cX.geluExactGrad() : cX.geluApproxGrad());
    }
}
