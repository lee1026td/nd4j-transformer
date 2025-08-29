package nn.transformer.embeddings;

import nn.core.Module;
import nn.core.Parameter;
import nn.initializer.Initializer;
import nn.optimizer.Optimizer;
import nn.transformer.mask.MaskUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import tensor.Tensor;

import java.util.Collections;
import java.util.List;

public class TokenEmbeddings implements Module {

    private final int vocabSize, d_model, padId;
    private final double scale;
    private final Parameter E;
    private final boolean isTrainable;
    private final double embDropProb;

    /* Caches */
    private int B, T;
    private int[] flatIds;
    private Tensor cEmbDrop;

    public TokenEmbeddings(int vocabSize, int d_model, int padId, Initializer eInit, boolean isTrainable, double embDropProb) {
        this.vocabSize = vocabSize; this.d_model = d_model; this.padId = padId;
        this.scale = Math.sqrt(d_model);
        this.isTrainable = isTrainable;
        this.embDropProb = embDropProb;

        this.E = new Parameter("token_embeddings.param", eInit.init(d_model, vocabSize), isTrainable);
    }

    @Override
    public Tensor forward(Tensor X, boolean training) {
        if(X.ndim() != 2) throw new IllegalArgumentException("TokenEmbeddings.forward expects ids of shape [B, T]");

        this.B = X.size(0);
        this.T = X.size(1);

        // Flatten ids -> int[]
        Tensor flattedX = X.reshape('c', B * T);

        INDArray flat = flattedX.getNDArray();

        int[] ids;

        if(flat.dataType().isIntType()) ids = flat.toIntVector();
        else {
            ids = new int[B * T];
            for(int i=0;i<ids.length;i++) {
                ids[i] = (int) flat.getDouble(i);
            }
        }
        this.flatIds = ids;

        // Assign table's row for each token : [B*T, d]
        INDArray rows = Nd4j.pullRows(E.getData().getNDArray(), 0, ids);

        // Scale by sqrt(d_model)
        rows.muli(scale);

        // Reshape to original shape : [B, T, d]
        Tensor out = new Tensor(rows.reshape('c', B, T, d_model));

        if(training) {
            cEmbDrop = MaskUtils.dropoutMaskLike(out, embDropProb);
            out = out.mul(cEmbDrop);
        } else cEmbDrop = null;

        return out;
    }

    // dY : [B, T, d] -> Accumulates dE
    // return : There are no prev layers -> return [B, T] zeros
    @Override
    public Tensor calcGradients(Tensor dY, boolean accumulate, double scale) {

        if(cEmbDrop != null) dY = dY.mul(cEmbDrop);

        INDArray dYArr = dY.getNDArray().reshape('c', B * T, d_model);  // [B*T, d]

        // 3) grad 버퍼 준비: gNew:[d, V] (항상 새 버퍼에 누적 → 마지막에 set/add)
        Tensor Edata = E.getData();
        if (Edata.size(0) != d_model) {
            throw new IllegalStateException("Embedding shape mismatch: E is [d,V] with d="
                    + Edata.size(0) + " but d_model=" + d_model);
        }

        Tensor gNewTensor = Tensor.zeros(Edata.shape());
        INDArray gNew = gNewTensor.getNDArray();

        // 4) forward에서 사용한 스케일(있다면) 반영
        final double s = scale * this.scale; // this.scale 예: sqrt(d_model)

        // 5) scatter-add: 각 토큰 id 열에 dY를 누적 (PAD 제외)
        for (int i = 0; i < flatIds.length; i++) {
            int id = flatIds[i];
            if (id == padId) continue; // PAD는 스킵

            // src: dY2[i,:]  -> [1,d] → [d,1] (열벡터). mul(s)는 새 배열을 만듭니다(원본 오염 방지).
            INDArray srcRow = dYArr.get(
                    NDArrayIndex.point(i),
                    NDArrayIndex.all());             // [1, d]
            INDArray srcCol = srcRow.reshape(d_model, 1);                      // [d, 1]
            if (s != 1.0) srcCol = srcCol.mul(s);                              // [d, 1]

            // 대상 열 뷰: rank-2 [d,1]로 유지하려면 'interval'을 사용!
            INDArray colView = gNew.get(
                    NDArrayIndex.all(),
                    NDArrayIndex.interval(id, id + 1)); // [d, 1]

            // in-place 누적
            colView.addi(srcCol); // [d,1] += [d,1]
        }

        // 6) grad 적용
        if (accumulate && E.getGrad() != null) {
            E.addGrad(new Tensor(gNew)); // 기존 grad += gNew
        } else {
            E.setGrad(new Tensor(gNew)); // grad = gNew
        }

        // no prev layers : return zeros
        return Tensor.zeros(B, T);
    }

    @Override
    public void update(Optimizer optimizer) {
        if(isTrainable) optimizer.step(E);
    }

    @Override
    public List<Parameter> parameters() {
        return Collections.singletonList(E);
    }

    @Override
    public void zeroGrad() {
        E.zeroGrad();
    }
}
