package nn.optimizer;

import nn.core.Parameter;
import tensor.Tensor;

import java.util.HashMap;
import java.util.Map;

public class StochasticGradientDescent implements Optimizer {

    private final double lr;
    private final double momentum;
    private final Map<Parameter, Tensor> momentumBuffer;

    public StochasticGradientDescent(double lr, double momentum) {
        this.lr = lr;
        this.momentum = momentum;
        momentumBuffer = new HashMap<>();
    }

    public StochasticGradientDescent(double lr) {
        this(lr, 0.0);
    }

    @Override
    public void step(Parameter p) {
        if(p.isTrainable()) {
            Tensor theta = p.getData();
            Tensor grad = p.getGrad();

            if(momentum == 0.0) {
                // SGD : param = param - lr * grad
                p.setData(theta.sub(grad.mul(lr)));

            } else {
                // SGD with Momentum
                Tensor vel = momentumBuffer.computeIfAbsent(
                        p, k -> Tensor.zeros(p.getData().shape())
                );

                // vel = momentum * v + grad
                vel = vel.mul(momentum).add(p.getGrad());
                momentumBuffer.put(p, vel);

                // param = param - lr * v
                p.setData(theta.sub(vel.mul(lr)));
            }
        }
    }
}
