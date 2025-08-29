package nn.activation;

import tensor.Tensor;

public class ELU implements Activation {

    private final double alpha;
    private Tensor cX;
    private Tensor cExp;

    public ELU(double alpha) {
        this.alpha = alpha;
    }

    @Override
    public Tensor forward(Tensor X) {
        this.cX = X;
        this.cExp = X.exp();
        return X.gt(0.0).mul(X).add(X.le(0.0).mul(cExp.sub(1.0).mul(alpha)));
    }

    @Override
    public Tensor backward(Tensor dY) {
        return dY.mul(cX.lt(0.0).mul(cExp.mul(alpha)).add(cX.ge(0.0)));
    }
}
