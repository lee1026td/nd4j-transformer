package nn.core;

import nn.optimizer.Optimizer;
import tensor.Tensor;

import java.util.List;

public interface Module {

    // Feed-forward (Tensor -> Tensor)
    Tensor forward(Tensor X, boolean training);

    // Calculates gradients
    Tensor calcGradients(Tensor dY, boolean accumulate, double scale);

    // Back-propagation (Tensor -> Tensor)
    default Tensor backward(Tensor dY) {
        return calcGradients(dY, false, 1.0);
    }

    // Back-propagation with gradient accumulation
    default Tensor backwardAccumulate(Tensor dY, double scale) {
        return calcGradients(dY, true, scale);
    }

    /* Multi-input API */
    default Tensor forwardMany(boolean training, Tensor... inputs) {
        int n = (inputs == null) ? 0 : inputs.length;
        if(n != 1) throw new IllegalArgumentException("1 input expected");

        return forward(inputs[0], training);
    }

    default Tensor[] calcGradientsMany(Tensor dY, boolean accumulate, double scale) {
        return new Tensor[]{calcGradients(dY, accumulate, scale)};
    }

    default Tensor[] backwardMany(Tensor dY) {
        return calcGradientsMany(dY, false, 1.0);
    }

    default Tensor[] backwardAccumulateMany(Tensor dY, double scale) {
        return calcGradientsMany(dY, true, scale);
    }

    // Optimizer : parameter update
    void update(Optimizer optimizer);

    // Get parameters list
    List<Parameter> parameters();

    // set gradient buffers to 0
    void zeroGrad();

}
