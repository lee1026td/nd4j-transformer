package nn.optimizer;

import nn.core.Parameter;
import tensor.Tensor;

import java.util.HashMap;
import java.util.Map;

public class Adam implements Optimizer {

    private final double lr;
    private final double beta1, beta2;
    private final double eps;

    private final Map<Parameter, Tensor> m;     // 1st moment
    private final Map<Parameter, Tensor> v;     // 2nd moment

    private final Map<Parameter, Integer> timeSteps;

    public Adam(double lr, double beta1, double beta2, double eps) {
        this.lr = lr;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.eps = eps;

        this.m = new HashMap<>();
        this.v = new HashMap<>();
        this.timeSteps = new HashMap<>();
    }

    public Adam(double lr, double beta1, double beta2) {
        this(lr, beta1, beta2, 1e-8);
    }

    @Override
    public void step(Parameter p) {
        if(p.isTrainable()) {
            // timestep initialize (or increment by 1)
            int t = timeSteps.getOrDefault(p, 0) + 1;
            timeSteps.put(p, t);

            // Param value
            Tensor theta = p.getData();
            Tensor grad = p.getGrad();

            if(grad == null) return;

            // Moment buffer initialization
            Tensor m_t = m.computeIfAbsent(p, k -> Tensor.zeros(theta.shape()));
            Tensor v_t = v.computeIfAbsent(p, k -> Tensor.zeros(theta.shape()));

            // Moment update
            // m_t = m_{t-1} * beta1 + (1 - beta1) * grad
            // v_t = v_{t-1} * beta2 + (1 - beta2) * grad^2
            m_t = m_t.mul(beta1).add(grad.mul(1.0 - beta1));
            v_t = v_t.mul(beta2).add(grad.pow(2).mul(1.0 - beta2));

            // Bias correction
            // mhat = m_t / (1 - beta1)
            // vhat = v_t / (1 - beta2)
            Tensor mhat = m_t.div(1.0 - beta1);
            Tensor vhat = v_t.div(1.0 - beta2);

            // Update param
            // theta_t = theta_{t-1} - lr * (mhat / (sqrt(vhat) + eps))
            Tensor denom = vhat.sqrt().add(eps);
            p.setData(theta.sub(mhat.mul(lr).div(denom)));

            // Save m, v
            m.put(p, m_t);
            v.put(p, v_t);
        }
    }
}
