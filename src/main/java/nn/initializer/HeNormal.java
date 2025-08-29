package nn.initializer;

import tensor.Tensor;

public class HeNormal implements Initializer {
    @Override
    public Tensor init(int inFeatuers, int outFeatures) {
        double std = Math.sqrt(2.0 / inFeatuers);
        return Tensor.randn(inFeatuers, outFeatures).mul(std);
    }
}
