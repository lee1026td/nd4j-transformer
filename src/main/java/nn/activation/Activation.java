package nn.activation;

import tensor.Tensor;

public interface Activation {

    Tensor forward(Tensor X);

    Tensor backward(Tensor dY);

}
