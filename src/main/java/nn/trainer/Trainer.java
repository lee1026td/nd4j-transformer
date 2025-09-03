package nn.trainer;

import nn.core.loss.Loss;
import nn.core.optimizer.Optimizer;
import nn.models.transformer.Transformer;

import tensor.Tensor;

import java.util.Random;

public final class Trainer {

    private final Transformer model;
    private final Optimizer opt;
    private final int V;
    private final int PAD;

    private final Random rnd = new Random(7);
    private final Loss lossFunc;

    public Trainer(Transformer model, Optimizer opt, Loss lossFunc, int vocabSize, int PADid) {
        this.model = model;
        this.opt   = opt;
        this.lossFunc = lossFunc;
        this.V     = vocabSize;
        this.PAD   = PADid;
    }

    public double[] trainEpoch(ToySeq2SeqDataset ds, int batchSize) {
        int N = ds.B;       // Total data num
        int S = ds.Smax;    // decoder sequence length
        int steps = (N + batchSize - 1) / batchSize;

        // Shuffle
        int[] order = new int[N];
        for(int i=0;i<N;i++) order[i] = i;
        for(int i=N-1;i>0;--i) {
            int j = rnd.nextInt(i + 1);
            int tmp = order[i]; order[i] = order[j]; order[j] = tmp;
        }

        double lossSum = 0.0;
        int correct = 0, total = 0;

        for(int step=0;step<steps;step++) {
            int start = step * batchSize;
            int B = Math.min(batchSize, N - start);

            // Batch
            Tensor srcIds = ds.batchSrc(start, B);      // [B, S]
            Tensor tgtInIds = ds.batchTgtIn(start, B);  // [B, S]

            Tensor tgtOutIds = ds.batchTgtOut(start, B);// [B, S]

            // Forward
            Tensor logits = model.forward(srcIds, tgtInIds, true);


            // Loss
            // PAD Masking for computing loss (w/o PAD tokens)
            Tensor padMask = tgtOutIds.ne(PAD);

            double lossVal = lossFunc.forward(logits, tgtOutIds, padMask);
            Tensor dLogits = lossFunc.backward();
            lossSum += lossVal;

            // Backward -> there are no returns in backward()
            model.backward(dLogits, false, 1.0);
            model.update(opt);
            model.zeroGrad();

            // Token Accuracy (ignores PAD)
            Tensor pred = logits.argmax(-1);    // [B, S]
            for(int i=0;i<B;i++) {
                for(int j=0;j<S;j++) {
                    int y = tgtOutIds.getInt(i, j);

                    if(y == PAD) continue;
                    if(pred.getInt(i, j) == y) correct++;
                    total++;
                }
            }
        }

        double avgLoss = lossSum / Math.max(1, steps);
        double tokenAcc = (total == 0) ? 0.0 : (double) correct / total;

        return new double[]{avgLoss, tokenAcc};
    }

}