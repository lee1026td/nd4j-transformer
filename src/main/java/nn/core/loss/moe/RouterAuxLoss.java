package nn.core.loss.moe;

import nn.layers.ffn.moe.MoERouter;
import tensor.Tensor;

public interface RouterAuxLoss {

    double forward(MoERouter router);

    Tensor backward(double scale);
}
