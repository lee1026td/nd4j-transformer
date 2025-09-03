package tensor;

import net.ericaro.neoitertools.Pair;
import nn.core.initializer.Initializer;
import org.nd4j.linalg.activations.impl.ActivationGELU;
import org.nd4j.linalg.activations.impl.ActivationSoftmax;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.TopK;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Arrays;

public class Tensor implements ITensor {

    private final INDArray data;

    public Tensor(INDArray data) {
        this.data = data;
    }

    public static Tensor zeros(int... shape) {
        return new Tensor(Nd4j.zeros(shape));
    }

    public static Tensor ones(int... shape) {
        return new Tensor(Nd4j.ones(shape));
    }

    public static Tensor fill(double value, int... shape) {
        return new Tensor(Nd4j.ones(shape).mul(value));
    }

    public static Tensor rand(int... shape) {
        return new Tensor(Nd4j.rand(shape));
    }

    public static Tensor randn(int... shape) {
        return new Tensor(Nd4j.randn(shape));
    }

    public static Tensor randomBernoulli(double p, int... shape) {
        long[] longShape = Arrays.stream(shape).asLongStream().toArray();

        return new Tensor(Nd4j.randomBernoulli(p, longShape));
    }

    public static Tensor from(double[] data, int... shape) {
        return new Tensor(Nd4j.createFromArray(data).reshape(shape));
    }

    public static Tensor from(double[][] data) {
        return new Tensor(Nd4j.createFromArray(data));
    }

    public static Tensor tile(Tensor tile, int... repeat) {
        return new Tensor(Nd4j.tile(tile.data, repeat));
    }

    public static Tensor concat(int dim, Tensor... tensors) {
        if (tensors == null || tensors.length == 0)
            throw new IllegalArgumentException("concat: at least one tensor required");
        int r = tensors[0].ndim();
        int axis = (dim < 0) ? dim + r : dim;

        // ND4J에 그대로 위임 (shape/rank 불일치 시 ND4J가 예외 던짐)
        org.nd4j.linalg.api.ndarray.INDArray[] arrs =
                new org.nd4j.linalg.api.ndarray.INDArray[tensors.length];
        for (int i = 0; i < tensors.length; i++) {
            if (tensors[i] == null) throw new IllegalArgumentException("concat: null tensor at " + i);
            arrs[i] = tensors[i].getNDArray(); // 또는 tensors[i].data
        }
        return new Tensor(org.nd4j.linalg.factory.Nd4j.concat(axis, arrs));
    }

    public static Tensor arange(int start, int end) {
        return new Tensor(Nd4j.arange(start, end).castTo(DataType.INT8));
    }

    @Override
    public int ndim() {
        return data.rank();
    }

    @Override
    public int size() {
        return (int) data.length();
    }

    @Override
    public int size(int dimension) {
        return (int) data.size(dimension);
    }

    @Override
    public int[] shape() {
        return Arrays.stream(data.shape()).mapToInt(l -> (int) l).toArray();
    }

    @Override
    public Tensor reshape(char order, int... newShape) {
        return new Tensor(data.reshape(order, newShape));
    }

    @Override
    public Tensor reshape(int... newShape) {
        return new Tensor(data.reshape(newShape));
    }


    @Override
    public Tensor permute(int... dims) {
        return new Tensor(data.permute(dims));
    }

    @Override
    public Tensor transpose() {
        return new Tensor(data.transpose());
    }

    @Override
    public Tensor transpose(int... dims) {
        int r = this.ndim();

        // no args : swap last two dim
        if(dims == null || dims.length == 0) {
            if(r < 2) return this;
            int[] perm = defaultPerm(r);
            swap(perm, r-2, r-1);
            return this.permute(perm);
        }

        // normalize negative dims
        for(int i=0;i<dims.length;i++) dims[i] = normalizeAxis(dims[i], r);

        // 2 args
        if(dims.length == 2) {
            int a = dims[0], b = dims[1];
            if(a == b) return this;
            int[] perm = defaultPerm(r);
            swap(perm, a, b);
            return this.permute(perm);
        }

        // number of args == rank
        if(dims.length == r)  {
            validatePermutation(dims, r);
            return this.permute(dims);
        }

        throw new IllegalArgumentException("Unexpected dims");
    }

    @Override
    public Tensor expandDims(int axis) {
        return new Tensor(Nd4j.expandDims(data, axis));
    }

    @Override
    public Tensor squeeze(int axis) {
        return new Tensor(Nd4j.squeeze(data, axis));
    }

    @Override
    public Tensor argmax(int axis) {
        return new Tensor(data.argMax(axis));
    }

    @Override
    public Tensor add(Tensor other) {
        return new Tensor(data.add(other.data));
    }

    @Override
    public Tensor sub(Tensor other) {
        return new Tensor(data.sub(other.data));
    }

    @Override
    public Tensor mul(Tensor other) {
        return new Tensor(data.mul(other.data));
    }

    @Override
    public Tensor div(Tensor other) {
        return new Tensor(data.div(other.data));
    }

    @Override
    public Tensor add(double scalar) {
        return new Tensor(data.add(scalar));
    }

    @Override
    public Tensor sub(double scalar) {
        return new Tensor(data.sub(scalar));
    }

    @Override
    public Tensor mul(double scalar) {
        return new Tensor(data.mul(scalar));
    }

    @Override
    public Tensor div(double scalar) {
        return new Tensor(data.div(scalar));
    }

    @Override
    public Tensor matmul(Tensor other) {
        INDArray A = this.data;
        INDArray B = other.data;

        if(this.ndim() == 2 && other.ndim() == 2) {
            INDArray C = Nd4j.createUninitialized(new long[]{size(0), other.size(1)}, 'c');
            Nd4j.gemm(A, B, C, false, false, 1.0, 0.0);
            return new Tensor(C);
        }
        else if(this.ndim() >= 3 && other.ndim() == 2) {
            return BatchedOps.mmulND2(this, other);
        }
        else if(this.ndim() >= 3 && other.ndim() >= 3) {
            return BatchedOps.bmmulND2(this, other);
        }
        else {
            throw new IllegalArgumentException("Matmul shape mismatch");
        }
    }

    @Override
    public Tensor sum(int axis, boolean keepDims) {
        return new Tensor(data.sum(keepDims, axis));
    }

    @Override
    public Tensor sum(boolean keepDims, int... axes) {
        return new Tensor(data.sum(keepDims, axes));
    }

    @Override
    public Tensor mean(int axis, boolean keepDims) {
        return new Tensor(data.mean(keepDims, axis));
    }

    @Override
    public Tensor var(int axis, boolean keepDims) {
        Tensor centered = this.sub(mean(axis, keepDims));
        Tensor v = centered.pow(2.0).mean(axis, keepDims);

        return v;
    }

    @Override
    public Tensor max(int axis, boolean keepDims) {
        return new Tensor(data.max(keepDims, axis));
    }

    @Override
    public Tensor min(int axis, boolean keepDims) {
        return new Tensor(data.min(keepDims, axis));
    }

    @Override
    public Tensor[] topK(int topK, int axis, boolean keepDims, boolean sorted) {

        INDArray[] sortedPair = Nd4j.sortWithIndices(this.data, axis, false);
        Tensor sortedInd = new Tensor(sortedPair[0]);
        Tensor sortedVals = new Tensor(sortedPair[1]);

        sortedInd = sortedInd.slice(axis, 0, topK);
        sortedVals = sortedVals.slice(axis, 0, topK);

        return new Tensor[]{ sortedVals, sortedInd };
    }

    @Override
    public Tensor exp() {
        return new Tensor(Nd4j.math().exp(data));
    }

    @Override
    public Tensor log() {
        return new Tensor(Nd4j.math().log(data));
    }

    @Override
    public Tensor sqrt() {
        return new Tensor(Nd4j.math().sqrt(data));
    }

    @Override
    public Tensor pow(double d) {
        return new Tensor(Nd4j.math().pow(data, d));
    }

    @Override
    public Tensor neg() {
        return new Tensor(data.neg());
    }

    @Override
    public Tensor reciprocal() {
        return new Tensor(Nd4j.math().reciprocal(data));
    }

    @Override
    public Tensor ge(double d) {
        double[] mask = data.gte(d).toDoubleVector();
        return Tensor.from(mask, this.shape());
    }

    @Override
    public Tensor gt(double d) {
        double[] mask = data.gt(d).toDoubleVector();
        return Tensor.from(mask, this.shape());
    }

    @Override
    public Tensor le(double d) {
        double[] mask = data.lte(d).toDoubleVector();
        return Tensor.from(mask, this.shape());
    }

    @Override
    public Tensor lt(double d) {
        return new Tensor(data.lt(d).castTo(DataType.FLOAT));
    }

    @Override
    public Tensor eq(double d) {
        return new Tensor(data.eq(d).castTo(DataType.FLOAT));
    }

    @Override
    public Tensor eq(Tensor other) {
        return new Tensor(data.eq(other.getNDArray()).castTo(DataType.FLOAT));
    }

    @Override
    public Tensor ne(double d) {
        return new Tensor(data.neq(d).castTo(DataType.FLOAT));
    }

    @Override
    public Tensor relu() {
        return new Tensor(Transforms.relu(data));
    }

    @Override
    public Tensor leakyRelu(double alpha) {
        return new Tensor(Transforms.leakyRelu(data));
    }

    @Override
    public Tensor sigmoid() {
        return new Tensor(Transforms.sigmoid(data));
    }

    @Override
    public Tensor tanh() {
        return new Tensor(Transforms.tanh(data));
    }

    @Override
    public Tensor geluApprox() {
        return new Tensor(new ActivationGELU(false).getActivation(data, false));
    }

    @Override
    public Tensor geluExact() {
        return new Tensor(new ActivationGELU(true).getActivation(data, false));
    }

    @Override
    public Tensor geluApproxGrad() {
        return new Tensor(new ActivationGELU(false)
                .backprop(data, Tensor.fill(1e-30, this.shape()).data)
                .getFirst()
        );
    }

    @Override
    public Tensor geluExactGrad() {
        return new Tensor(new ActivationGELU(true)
                .backprop(data, Tensor.fill(1e-30, this.shape()).data)
                .getFirst()
        );
    }

    @Override
    public Tensor softmax() {
        return new Tensor(new ActivationSoftmax().getActivation(data, false));
    }

    @Override
    public void set(double val, int... indices) {
        data.putScalar(indices, val);
    }

    @Override
    public Tensor get(int... indices) {
        return new Tensor(data.get(Arrays.stream(indices).mapToObj(NDArrayIndex::point).toArray(INDArrayIndex[]::new)));
    }

    @Override
    public Tensor slice(int dim, int start, int end) {
        INDArrayIndex[] idx = new INDArrayIndex[data.rank()];
        for(int i = 0; i< data.rank(); i++) idx[i] = NDArrayIndex.all();
        idx[normalizeAxis(dim, data.rank())] = NDArrayIndex.interval(start, end);


        INDArray view = data.get(idx).dup('c');
        return new Tensor(view);
    }

    @Override
    public double getDouble(int... indices) {
        return data.getDouble(indices);
    }

    @Override
    public int getInt(int... indices) {
        return data.getInt(indices);
    }

    @Override
    public double[] toDoubleArray() {
        return data.ravel().toDoubleVector();
    }

    @Override
    public INDArray getNDArray() {
        return data;
    }

    @Override
    public Tensor dup() {
        return new Tensor(data.dup());
    }

    @Override
    public String toString() {
        return "Tensor " + getShapeToString() + ": \n" + data.toString() + "\n";
    }

    public String getShapeToString() {
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        for(int i=0;i<this.ndim();i++) {
            sb.append(size(i));
            if(i < this.ndim() - 1) sb.append(", ");
        }
        sb.append("]");

        return sb.toString();
    }

    /* private helper methods */

    private static int[] defaultPerm(int r) {
        int[] p = new int[r];
        for(int i=0;i<r;i++) p[i] = i;
        return p;
    }

    private static void swap(int[] a, int i, int j) {
        int t = a[i]; a[i] = a[j]; a[j] = t;
    }

    private static int normalizeAxis(int ax, int r) {
        if(ax < 0) ax += r;
        if(ax < 0 || ax >= r)
            throw new IllegalArgumentException("axis out of range : " + ax + " for rank " + r);
        return ax;
    }

    public static void validatePermutation(int[] perm, int r) {
        boolean[] seen = new boolean[r];
        for(int v : perm) {
            if(v < 0 || v >= r || seen[v]) throw new IllegalArgumentException("invalid permutation");
            seen[v] = true;
        }
    }
}
