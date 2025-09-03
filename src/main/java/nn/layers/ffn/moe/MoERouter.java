package nn.layers.ffn.moe;

import nn.core.Module;
import nn.core.Parameter;
import nn.core.activation.Softplus;
import nn.core.initializer.Initializer;
import nn.core.optimizer.Optimizer;
import nn.layers.Linear;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.strict.SoftPlus;
import org.nd4j.linalg.factory.Nd4j;
import tensor.Tensor;

import java.util.List;

public class MoERouter implements Module {

    private final Linear projLogits;
    private final Linear projNoise;

    private final int d_model;
    private final int numExperts;
    private final int topK;
    private final boolean useNoisyTopK;
    private final double capacityFactor;

    private final boolean isTrainable;

    private Tensor X;

    public MoERouter(int d_model, int numExperts, int topK, boolean useNoisyTopK, double capacityFactor, Initializer wInit, boolean isTrainable) {
        this.d_model = d_model;
        this.numExperts = numExperts;
        this.topK = topK;
        this.useNoisyTopK = useNoisyTopK;
        this.capacityFactor = capacityFactor;

        this.isTrainable = isTrainable;

        this.projLogits = new Linear(d_model, numExperts, wInit, isTrainable);
        if(useNoisyTopK) this.projNoise = new Linear(d_model, numExperts, wInit, isTrainable);
        else this.projNoise = null;
    }

    @Override
    public Tensor forward(Tensor X, boolean training) {
        // Reshape : [B, T, d] -> [B*T, d]
        int B = X.size(0);
        int T = X.size(1);
        //X = X.reshape(B*T, d_model);
        this.X = X;

        // Compute logits for after softmax
        Tensor logits = projLogits.forward(X, training);        // [B, T, numExperts]
        // For Noisy Top-K routing
        if(useNoisyTopK) {
            Tensor noise = new Softplus().forward(projNoise.forward(X, training));
            noise = noise.mul(Tensor.randn(noise.shape()));
            logits = logits.add(noise);
        }

        // Top-K expert selection : compute probs over active experts
        Tensor[] topKOut = logits.topK(topK, -1, true, true);
        Tensor topKLogits = topKOut[0];     // [B, T, topK]
        Tensor topKIndices = topKOut[1];
        // Softmax : [B, T, numExperts], set non-top-k vals to -inf
        Tensor probs = Tensor.fill(Float.NEGATIVE_INFINITY, topKLogits.shape());
        for(int i=0;i<B;i++) {
            for(int j=0;j<T;j++) {
                for(int k=0;k<topK;k++) {
                    int validIndex = topKIndices.getInt(i, j, k);
                    double topKval = topKLogits.getDouble(i, j, k);

                    probs.set(topKval, i, j, validIndex);
                }
            }
        }

        probs = probs.softmax();




        return null;
    }

    @Override
    public Tensor calcGradients(Tensor dY, boolean accumulate, double scale) {
        return null;
    }

    @Override
    public void update(Optimizer optimizer) {

    }

    @Override
    public List<Parameter> parameters() {
        return List.of();
    }

    @Override
    public void zeroGrad() {

    }
}
