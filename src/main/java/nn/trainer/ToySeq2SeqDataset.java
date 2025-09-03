package nn.trainer;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import tensor.Tensor;

import java.util.Random;

/** 간단한 시퀀스-시퀀스 데이터셋: copy 또는 reverse */
public final class ToySeq2SeqDataset {
    public enum Task { COPY, REVERSE }

    public final int B;         // 총 샘플 수
    public final int Smax;      // 최대 src 길이
    public final int T;      // 최대 tgt 길이(= Smax + 1 for BOS/EOS 포함 시)
    public final int V;         // vocab size
    public final int PAD, BOS, EOS;

    public final int[][] src;       // [B, Smax]  (우측 PAD)
    public final int[][] tgtIn;     // [B, Tmax]  [BOS, y..., PAD...]
    public final int[][] tgtOut;    // [B, Tmax]  [y..., EOS, PAD...]

    private ToySeq2SeqDataset(int B, int Smax, int V, int PAD, int BOS, int EOS) {
        this.B = B; this.Smax = Smax; this.T = Smax; // y... + EOS
        this.V = V; this.PAD = PAD; this.BOS = BOS; this.EOS = EOS;
        this.src = new int[B][Smax];
        this.tgtIn = new int[B][T];
        this.tgtOut = new int[B][T];
    }

    public static ToySeq2SeqDataset make(Task task, int B, int Smax, int V, int PAD, int BOS, int EOS) {
        if (V < 4) throw new IllegalArgumentException("Vocab must be >= 4");

        Random rnd = new Random();
        ToySeq2SeqDataset ds = new ToySeq2SeqDataset(B, Smax, V, PAD, BOS, EOS);

        for (int b = 0; b < B; b++) {
            int L = 1 + rnd.nextInt(Smax); // 1..Smax
            int[] x = new int[L];
            for (int i = 0; i < L; i++) x[i] = 3 + rnd.nextInt(V - 3);

            // src: [x..., PAD...]
            for (int t = 0; t < ds.Smax; t++) ds.src[b][t] = (t < L) ? x[t] : PAD;

            // target core y
            int[] y;
            if (task == Task.COPY) {
                y = x;
            } else {
                y = new int[L];
                for (int i = 0; i < L; i++) y[i] = x[L - 1 - i];
            }

            // ---- 디코더 입력/라벨 길이: T = Smax ----
            ds.tgtIn[b][0] = BOS;

            if (L < ds.T) {
                // tgtIn: [BOS, y[0..L-1], PAD...]
                for (int t = 1; t <= L; t++) ds.tgtIn[b][t] = y[t - 1];
                for (int t = L + 1; t < ds.T; t++) ds.tgtIn[b][t] = PAD;

                // tgtOut: [y[0..L-1], EOS, PAD...]
                for (int t = 0; t < L; t++) ds.tgtOut[b][t] = y[t];
                ds.tgtOut[b][L] = EOS;
                for (int t = L + 1; t < ds.T; t++) ds.tgtOut[b][t] = PAD;

            } else { // L == T(=Smax) → EOS 자리를 만들기 위해 마지막 y 하나 truncate
                // tgtIn: [BOS, y[0..T-2]]
                for (int t = 1; t < ds.T; t++) ds.tgtIn[b][t] = y[t - 1]; // y[0..T-2]
                // tgtOut: [y[0..T-2], EOS]
                for (int t = 0; t < ds.T - 1; t++) ds.tgtOut[b][t] = y[t]; // y[0..T-2]
                ds.tgtOut[b][ds.T - 1] = EOS;
            }
        }
        return ds;
    }

    // 배치 텐서로 변환
    public Tensor batchSrc(int start, int batchSize) {
        INDArray a = Nd4j.create(batchSize, Smax);
        for (int i = 0; i < batchSize; i++) {
            int b = start + i;
            for (int t = 0; t < Smax; t++) a.putScalar(i, t, src[b][t]);
        }
        return new Tensor(a);
    }
    public Tensor batchTgtIn(int start, int batchSize) {
        INDArray a = Nd4j.create(batchSize, T);
        for (int i = 0; i < batchSize; i++) {
            int b = start + i;
            for (int t = 0; t < T; t++) a.putScalar(i, t, tgtIn[b][t]);
        }
        return new Tensor(a);
    }
    public Tensor batchTgtOut(int start, int batchSize) {
        INDArray a = Nd4j.create(batchSize, T);
        for (int i = 0; i < batchSize; i++) {
            int b = start + i;
            for(int t = 0; t < T; t++) a.putScalar(i, t, tgtOut[b][t]);
        }
        return new Tensor(a);
    }
}
