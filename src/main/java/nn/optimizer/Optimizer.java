package nn.optimizer;

import nn.core.Parameter;

public interface Optimizer {

    void step(Parameter p);
}
