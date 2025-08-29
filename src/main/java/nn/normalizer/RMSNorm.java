package nn.normalizer;

import nn.core.Module;
import nn.core.Parameter;
import nn.optimizer.Optimizer;
import org.nd4j.linalg.factory.Nd4j;
import tensor.Tensor;

import java.util.Arrays;
import java.util.List;


public class RMSNorm implements Module {

    private final double eps;
    private final int numFeatures;
    private final boolean isTrainable;

    // Parameter
    private Parameter gamma;

    // Cached
    private Tensor X;
    private Tensor invR;
    private Tensor xHat;

    public RMSNorm(int numFeatures, double eps, boolean isTrainable) {
        this.numFeatures = numFeatures;
        this.eps = eps;
        this.isTrainable = isTrainable;

        this.gamma = new Parameter("rmsnorm.gamma", Tensor.ones(numFeatures), isTrainable);
    }

    public RMSNorm(int numFeatures, double eps) {
        this(numFeatures, eps, true);
    }

    // RMS = sqrt((1/D) * sum(x_i^2))
    //   x_norm = x / (RMS + eps)
    //  y = gamma * x_norm


    @Override
    public Tensor forward(Tensor X, boolean training) {
        assert(X.shape()[X.ndim() - 1] == numFeatures) : "Number of features != Normalized size";

        this.X = X;

        // x^2 : [*, D]
        Tensor sq = X.pow(2.0);

        // Mean squared : [*, 1]
        Tensor meanSq = sq.mean(-1, true);

        // RMS : sqrt(meanSq + eps) : [*, 1]
        Tensor rms = meanSq.add(eps).sqrt();

        // Inverted RMS : [*, 1]
        Tensor invRms = rms.reciprocal();
        this.invR = invRms;

        // Normalization : x / RMS : [*, D]
        Tensor xHat = X.mul(invRms);
        this.xHat = xHat;

        // [D] broadcast : gamma * xHat -> [*, D]
        Tensor out = xHat.mul(gamma.getData());

        return out;
    }

    @Override
    public Tensor calcGradients(Tensor dY, boolean accumulate, double scale) {
        assert(X.shape()[X.ndim() - 1] == numFeatures) : "Number of features != Normalized size";

        if(scale != 1.0) dY = dY.mul(scale);

        // [B, N, D] -> [B*N, D]
        int B = dY.size() / numFeatures;

        // dGamma
        Tensor dGamma = dY.mul(xHat).reshape(B, numFeatures).sum(0, false);

        // dNorm
        Tensor dNorm = dY.mul(gamma.getData());

        // dRMS
        Tensor invR = this.invR;
        Tensor invR3 = invR.pow(3.0);

        Tensor term1 = dNorm.mul(invR);

        // scalar per row
        Tensor dot = dNorm.mul(X).sum(-1, true);
        Tensor scalarPerRow = dot.div(numFeatures);

        Tensor term2 = X.mul(scalarPerRow).mul(invR3);

        // dX
        Tensor dX = term1.sub(term2);

        if(accumulate) gamma.addGrad(dGamma);
        else gamma.setGrad(dGamma);

        return dX;
    }

    @Override
    public void update(Optimizer optimizer) {
        if(isTrainable) optimizer.step(gamma);
    }

    @Override
    public List<Parameter> parameters() {
        return List.of(gamma);
    }

    @Override
    public void zeroGrad() {
        if(gamma.getGrad() != null) gamma.zeroGrad();
    }
}
