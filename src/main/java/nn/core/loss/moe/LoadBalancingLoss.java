package nn.core.loss.moe;

import nn.layers.Linear;
import nn.layers.ffn.moe.MoERouter;
import tensor.Tensor;

public class LoadBalancingLoss implements RouterAuxLoss {

    private MoERouter router;

    @Override
    public double forward(MoERouter router) {
        this.router = router;
        Tensor fMean = router.getfMean();
        Tensor pMean = router.getpMean();

        int numExperts = router.getNumExperts();

        // LB-loss = alpha * N * sum(fMean * pMean)
        return fMean.mul(pMean).sum(-1, false).mul(numExperts).getDouble(0);
    }

    @Override
    public Tensor backward(double scale) {
        Tensor probs = router.getProbsAll();   // [B, T, N]

        int B = probs.size(0); int T = probs.size(1);
        Tensor fMean = router.getfMean();

        int topK = router.getTopK();
        int numExperts = router.getNumExperts();

        // [B, T, N]
        Tensor dP = Tensor.tile(fMean.reshape(1, 1, numExperts), B, T).mul((double) numExperts / B*T*topK);

        // Softmax gradient : dG = (dP - sum(dP * P)) * P -> [B, T, N]
        Tensor gy = dP.mul(probs).sum(-1, true);
        Tensor dG = dP.sub(gy).mul(probs);

        return dG;
    }
}
