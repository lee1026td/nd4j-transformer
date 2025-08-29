package nn.transformer;

import nn.transformer.trainer.ToySeq2SeqDataset;
import org.nd4j.linalg.factory.Nd4j;
import tensor.Tensor;

import java.util.Arrays;

public class Generator {

    public static Tensor greedyDecode(Transformer model,
                                      Tensor srcTokens, int maxLen,
                                      int bosId, int eosId, int padId) {

        double NEG_INF = -1e9;

        // valid source length
        int srcLen = validSequenceLength(srcTokens, padId);

        // Prevents empty sequence (while tLen <= minGen : no EOS)
        int minGen = Math.max(1, Math.min(srcLen, maxLen - 1));

        // Decoding buffer

        int[] out = new int[maxLen];
        Arrays.fill(out, padId);
        int tLen = 0;
        out[tLen++] = bosId;      // Start with single BOS token

        for(;tLen<maxLen;tLen++) {
            Tensor tgtIdsGen = new Tensor(Nd4j.createFromArray(out).reshape(1, maxLen));

            // Predict -> no dropout (training == false)
            Tensor logits = model.forward(srcTokens, tgtIdsGen, false);
            int V = logits.size(-1);

            // Last logit : [1, 1, V] -> [1, V]
            Tensor last = logits.slice(1, tLen - 1, tLen).reshape(1, V);

            // Banned tokens (BOS, PAD), while tLen <= minGen : no EOS
            Tensor classes = Tensor.arange(0, V).reshape(1, V);         // [1, V]
            Tensor ban = classes.eq(bosId).add(classes.eq(padId));

            if(tLen <= minGen) ban = ban.add(classes.eq(eosId));
            last = last.add(ban.mul(NEG_INF));

            // Greedy choosing
            int nextId = (int) last.argmax(1).toDoubleArray()[0];
            out[tLen] = nextId;

            if(nextId == eosId) {
                int[] trimmed = Arrays.copyOfRange(out, 1, tLen);   // no BOS/EOS
                return new Tensor(Nd4j.createFromArray(trimmed));
            }
        }

        // Reached maxLen : no BOS
        int[] trimmed = Arrays.copyOfRange(out, 1, tLen);
        return new Tensor(Nd4j.createFromArray(trimmed));
    }

    private static int validSequenceLength(Tensor sequence, int padId) {
        int validLen = 0;
        int S = sequence.size(-1);

        for(int i=0;i<S;i++) {
            if(sequence.getInt(0, i) == padId) break;
            validLen++;
        }

        return validLen;
    }
}
