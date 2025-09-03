package nn.core.initializer;

import tensor.Tensor;

public interface Initializer {

    Tensor init(int inFeatuers, int outFeatures);
}
