package nn.transformer.modules;

import nn.activation.Activation;
import nn.core.Module;
import nn.core.Parameter;
import nn.initializer.Initializer;
import nn.layers.Linear;
import nn.optimizer.Optimizer;
import nn.transformer.mask.MaskUtils;
import tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

public class FeedForwardNetwork implements Module {

    private final Linear l1, l2;
    private final Activation act;
    private final boolean isTrainable;

    private final double dropProb;

    private Tensor cDropAct, cDropOut;

    public FeedForwardNetwork(int d_model, int d_hidden, Activation act, Initializer wInit, Initializer bInit, boolean isTrainable, double dropProb) {
        this.act = act;
        this.isTrainable = isTrainable;

        this.dropProb = dropProb;

        this.l1 = new Linear(d_model, d_hidden, wInit, bInit, isTrainable);
        this.l2 = new Linear(d_hidden, d_model, wInit, bInit, isTrainable);
    }
    @Override
    public Tensor forward(Tensor X, boolean training) {
        Tensor Z1 = l1.forward(X, training);
        Tensor H1 = act.forward(Z1);

        if(training) {
            cDropAct = MaskUtils.dropoutMaskLike(H1, dropProb);
            H1 = H1.mul(cDropAct);
        } else {
            cDropAct = null;
        }

        Tensor Y = l2.forward(H1, training);

        if(training) {
            cDropOut = MaskUtils.dropoutMaskLike(Y, dropProb);
            Y = Y.mul(cDropOut);
        } else {
            cDropOut = null;
        }

        return Y;
    }

    @Override
    public Tensor calcGradients(Tensor dY, boolean accumulate, double scale) {
        Tensor dY_s = (scale == 1.0) ? dY : dY.mul(scale);

        if(cDropOut != null) dY_s = dY_s.mul(cDropOut);

        Tensor dH = l2.calcGradients(dY_s, accumulate, 1.0);

        if(cDropAct != null) dH = dH.mul(cDropAct);

        Tensor dZ1 = act.backward(dH);
        Tensor dX = l1.calcGradients(dZ1, accumulate, 1.0);

        return dX;
    }

    @Override
    public void update(Optimizer optimizer) {
        if(isTrainable) {
            l1.update(optimizer);
            l2.update(optimizer);
        }
    }

    @Override
    public List<Parameter> parameters() {
        List<Parameter> ps = new ArrayList<>();
        ps.addAll(l1.parameters());
        ps.addAll(l2.parameters());

        return ps;
    }

    @Override
    public void zeroGrad() {
        l1.zeroGrad(); l2.zeroGrad();
    }
}
