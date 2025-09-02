import nn.initializer.*;
import nn.layers.*;
import nn.loss.*;
import nn.normalizer.*;
import nn.optimizer.*;
import nn.transformer.*;
import nn.transformer.embeddings.*;
import nn.transformer.mask.MaskUtils;
import nn.transformer.modules.*;
import nn.transformer.trainer.*;
import tensor.Nd4jInit;
import tensor.Tensor;

import java.util.Arrays;
import java.util.Random;
import java.util.function.Supplier;

public class Main {
    public static void main(String[] args) {
        Nd4jInit.configure();

        int V = 200;
        int d_model = 128;
        int nHead = 2;
        int d_hidden = 512;
        int N = 3;

        double dropout = 0.1;
        double lnEps = 1e-5;
        int maxLen = 20;
        int BOS = 1, EOS = 2, PAD = 0;

        int trainN = 100;
        int validN = 10;
        int batchSize = 10;
        int epochs = 5;

        double lr = 5e-3;
        double beta1 = 0.9, beta2 = 0.98, adamEps = 1e-8;

        Supplier<LayerNorm> layerNormSupplier = () -> new LayerNorm(d_model, lnEps, true);

        TokenEmbeddings srcEmb = new TokenEmbeddings(V, d_model, PAD, new XavierNormal(), true, dropout);
        TokenEmbeddings tgtEmb = new TokenEmbeddings(V, d_model, PAD, new XavierNormal(), true, dropout);
        SinusoidalPositionalEncoding posEnc = new SinusoidalPositionalEncoding(maxLen, d_model);
        Encoder encoder = new Encoder(d_model, d_hidden, nHead, N, layerNormSupplier, true, dropout, dropout);
        Decoder decoder = new Decoder(d_model, d_hidden, nHead, N, layerNormSupplier, true, dropout, dropout);
        Linear lmHead = new Linear(d_model, V, new XavierNormal(), new HeNormal(), true, dropout);

        // Weight tying
        lmHead.setParameter(tgtEmb.parameters().get(0));

        // Model
        Transformer model = new Transformer(srcEmb, tgtEmb, posEnc,
                encoder, decoder, lmHead,
                PAD, BOS, EOS);

        Loss ceLoss = new CrossEntropyLoss();
        Optimizer opt = new Adam(lr, beta1, beta2, adamEps);

        ToySeq2SeqDataset trainData = ToySeq2SeqDataset.make(ToySeq2SeqDataset.Task.REVERSE, trainN, maxLen, V, PAD, BOS, EOS);
        ToySeq2SeqDataset validData = ToySeq2SeqDataset.make(ToySeq2SeqDataset.Task.REVERSE, validN, maxLen, V, PAD, BOS, EOS);

        Trainer trainer = new Trainer(model, opt, ceLoss, V, PAD);

        /* ===== 에폭 루프 ===== */

        Random r = new Random();

        for (int e = 1; e <= epochs; e++) {
            double[] trainRes = trainer.trainEpoch(trainData, batchSize);
            System.out.printf("epoch %d | loss=%.4f | tokenAcc=%.4f%n", e, trainRes[0], trainRes[1]);

            /* For validataion */

            /*
            int idx = r.nextInt(validN);
            Tensor valSrcId = validData.batchSrc(idx, 1);
            Tensor generated = Generator.greedyDecode(model, valSrcId, maxLen, BOS, EOS, PAD);

            System.out.println(valSrcId);
            System.out.println(generated);

             */
            if(trainRes[1] >= 0.95) break;
        }

        /* Validation */

        for(int i=0;i<validN;i++) {
            Tensor validSrcIn = validData.batchSrc(i, 1);
            Tensor generated = Generator.greedyDecode(model, validSrcIn, maxLen, BOS, EOS, PAD);
            System.out.println(validSrcIn);
            System.out.println(generated);
        }
    }
}
