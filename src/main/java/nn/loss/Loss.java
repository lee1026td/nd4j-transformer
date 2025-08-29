package nn.loss;

import tensor.Tensor;

public interface Loss {

    double forward(Tensor preds, Tensor tgts);

    double forward(Tensor preds, Tensor tgts, Tensor mask);

    Tensor backward();
}
