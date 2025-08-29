package nn.core;

import tensor.Tensor;

public class Parameter {

    private final String name;
    private Tensor data;
    private Tensor grad;

    private boolean trainable = true;

    public Parameter(String name, Tensor data, boolean trainable) {
        this.name = name;
        this.data = data;
        this.trainable = trainable;
    }

    public String getName() { return name; }
    public Tensor getData() { return data; }
    public Tensor getGrad() { return grad; }

    public void setData(Tensor data) {
        this.data = data;
    }

    public void setGrad(Tensor grad) {
        this.grad = grad;
    }

    public void setTrainable(boolean trainable) {
        this.trainable = trainable;
    }

    public boolean isTrainable() { return trainable; }

    public void zeroGrad() {
        grad = null;
    }

    public void addGrad(Tensor grad) {
        this.grad = (this.grad == null) ? grad : this.grad.add(grad);
    }
}
