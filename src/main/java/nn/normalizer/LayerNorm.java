package nn.normalizer;

import nn.core.Module;
import nn.core.Parameter;
import nn.optimizer.Optimizer;
import tensor.Tensor;

import java.util.List;

public class LayerNorm implements Module {

    private final int numFeatures;
    private final double eps;
    private final boolean isTrainable;

    /* Learnable Parameters */
    private Parameter gamma;
    private Parameter beta;

    /* Cached */
    private Tensor cX;
    private Tensor cMean;
    private Tensor cVar;
    private Tensor cNorm;

    public LayerNorm(int numFeatures, double eps, boolean isTrainable) {
        this.numFeatures = numFeatures;
        this.eps = eps;
        this.isTrainable = isTrainable;
        this.gamma = new Parameter("layernorm.gamma", Tensor.ones(numFeatures), isTrainable);
        this.beta = new Parameter("layernorm.beta", Tensor.zeros(numFeatures), isTrainable);
    }

    public LayerNorm(int numFeatures, double eps) {
        this(numFeatures, eps, true);
    }

    @Override
    public Tensor forward(Tensor X, boolean training) {
        assert (X.shape()[X.ndim() - 1] == numFeatures) : "Number of features != Normalized size";

        this.cX = X;

        // Mean
        Tensor mean = X.mean(-1, true);
        this.cMean = mean;

        // Centralize : x - mean
        Tensor centered = X.sub(mean);

        // Variance
        Tensor var = X.var(-1, true);
        this.cVar = var;

        // Normalize : (x - mean) / (std + eps)
        Tensor normalized = centered.div(var.add(eps).sqrt());
        this.cNorm = normalized;

        // Scale & Shift : gamma * x_norm + beta
        return normalized.mul(gamma.getData()).add(beta.getData());
    }

    /*
     *
     *  dL/dbeta = sum(dY)
     *  dL/dGamma = sum(dY * cNorm)
     *  dL/d(cNorm) = dY * gamma
     *  dL/d(cStd) = -sum(dY * gamma * (x - cMean)) / cVar
     *  dL/d(cMean) = -sum(dY * gamma) / std - (dL/d(cStd)) * sum(x - cMean) / (D * std)
     *
     *  dL/dX = (dY * gamma) / std + (-sum(dY * gamma * (x - cMean)) / cVar * sum(X - cMean) / D * std^3) + dL/d(cMean) / D
     *
     */

    @Override
    public Tensor calcGradients(Tensor dY, boolean accumulate, double scale) {
        assert(cX.shape()[cX.ndim() - 1] == numFeatures) : "Number of features != Normalized size";

        if(scale != 1.0) dY = dY.mul(scale);

        // [B, N, D] -> [B*N, D] for sum
        int B = dY.size() / numFeatures;

        // dBeta
        Tensor dBeta = dY.reshape(B, numFeatures).sum(0, false);

        // dGamma
        Tensor dGamma = dY.mul(cNorm).reshape(B, numFeatures).sum(0, false);

        // dNorm
        Tensor dNorm = dY.mul(gamma.getData());

        Tensor centered = cX.sub(cMean);
        Tensor std = cVar.add(eps).sqrt();

        // dStd
        Tensor dStd = dNorm.mul(centered).div(std).sum(-1, true).mul(1.0);

        // dMean
        Tensor dMean = dNorm.sum(-1, true)
                .div(std.mul(-1.0))
                .sub(dStd.mul(centered.sum(-1, true)).div(numFeatures).div(std));

        // dX
        Tensor dX = dNorm.div(std)
                .add(dStd.mul(centered).mul(2.0).div(numFeatures).div(std))
                .add(dMean.div(numFeatures));

        if(accumulate) {
            gamma.addGrad(dGamma); beta.addGrad(dBeta);
        } else {
            gamma.setGrad(dGamma); beta.setGrad(dBeta);
        }

        return dX;
    }

    @Override
    public void update(Optimizer optimizer) {
        if(isTrainable) {
            optimizer.step(gamma);
            optimizer.step(beta);
        }
    }

    @Override
    public List<Parameter> parameters() {
        return List.of(gamma, beta);
    }

    @Override
    public void zeroGrad() {
        if(gamma.getGrad() != null) gamma.zeroGrad();
        if(beta.getGrad() != null) beta.zeroGrad();
    }
}
