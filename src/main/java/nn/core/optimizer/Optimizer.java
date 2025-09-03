package nn.core.optimizer;

import nn.core.Parameter;

public interface Optimizer {

    void step(Parameter p);
}
