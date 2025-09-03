package tensor;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface ITensor {

    int ndim();
    int size();
    int size(int dimension);
    int[] shape();

    Tensor reshape(char order, int... newShape);
    Tensor reshape(int... newShape);
    Tensor permute(int... dims);
    Tensor transpose();
    Tensor transpose(int... dims);
    Tensor expandDims(int axis);
    Tensor squeeze(int axis);
    Tensor argmax(int axis);

    Tensor add(Tensor other);
    Tensor sub(Tensor other);
    Tensor mul(Tensor other);
    Tensor div(Tensor other);

    Tensor add(double scalar);
    Tensor sub(double scalar);
    Tensor mul(double scalar);
    Tensor div(double scalar);

    Tensor matmul(Tensor other);

    Tensor sum(int axis, boolean keepDims);
    Tensor sum(boolean keepDims, int... axes);
    Tensor mean(int axis, boolean keepDims);
    Tensor var(int axis, boolean keepDims);
    Tensor max(int axis, boolean keepDims);
    Tensor min(int axis, boolean keepDims);
    Tensor[] topK(int topK, int axis, boolean keepDims, boolean sorted);

    Tensor exp();
    Tensor log();
    Tensor sqrt();
    Tensor pow(double d);
    Tensor neg();
    Tensor reciprocal();

    Tensor ge(double d);
    Tensor gt(double d);
    Tensor le(double d);
    Tensor lt(double d);
    Tensor eq(double d);
    Tensor eq(Tensor other);
    Tensor ne(double d);

    Tensor relu();
    Tensor leakyRelu(double alpha);
    Tensor sigmoid();
    Tensor tanh();
    Tensor geluApprox();
    Tensor geluExact();
    Tensor geluApproxGrad();
    Tensor geluExactGrad();
    Tensor softmax();

    void set(double val, int... indices);
    Tensor get(int... indices);
    Tensor slice(int dim, int start, int end);
    double getDouble(int... indices);
    int getInt(int... indices);


    double[] toDoubleArray();
    INDArray getNDArray();
    Tensor dup();
}
