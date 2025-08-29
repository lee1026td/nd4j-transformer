package nn.initializer;

import tensor.ITensor;
import tensor.Tensor;

public class XavierNormal implements Initializer {
    @Override
    public Tensor init(int inFeatuers, int outFeatures) {
        double std = Math.sqrt(1.0 / (inFeatuers + outFeatures));
        return Tensor.randn(inFeatuers, outFeatures).mul(std);
    }
}
