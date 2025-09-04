import nn.core.activation.GELU;
import nn.core.initializer.*;
import nn.core.loss.moe.LoadBalancingLoss;
import nn.layers.*;
import nn.core.loss.*;
import nn.layers.ffn.moe.MoEFeedForward;
import nn.layers.ffn.moe.MoERouter;
import nn.layers.normalizer.*;
import nn.core.optimizer.*;
import nn.layers.embeddings.*;
import nn.mask.MaskUtils;
import nn.models.transformer.Generator;
import nn.models.transformer.Transformer;
import nn.modules.*;
import nn.trainer.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import tensor.Nd4jInit;
import tensor.Tensor;

import java.util.Random;
import java.util.function.Supplier;

public class Main {
    private static final Logger log = LoggerFactory.getLogger(Main.class);

    public static void main(String[] args) {
        Nd4jInit.configure();

        int B = 2;
        int T = 4;
        int d = 9;
        int N = 5;
        int K = 2;
        double cFactor = 1.0;

        Tensor X = Tensor.rand(B, T, d);
        System.out.println("X : \n" + X);

        MoEFeedForward moeFFN = new MoEFeedForward(d, d*4, N, K, true, cFactor, new GELU(true), new XavierNormal(), new HeNormal(), true, 0.1);

        Tensor Y = moeFFN.forward(X, true);

        System.out.println("Y : \n" + Y);

        MoERouter router = moeFFN.getRouter();

        LoadBalancingLoss lb = new LoadBalancingLoss();
        double lbloss = lb.forward(router);
        System.out.println(lbloss);

        /*

        int V = 200;
        int d_model = 128;
        int nHead = 2;
        int d_hidden = 256;
        int N = 2;

        double dropout = 0.0;
        double lnEps = 1e-5;
        int maxLen = 20;
        int BOS = 1, EOS = 2, PAD = 0;

        int trainN = 1000;
        int validN = 10;
        int batchSize = 10;
        int epochs = 500;

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

        ToySeq2SeqDataset trainData = ToySeq2SeqDataset.make(ToySeq2SeqDataset.Task.COPY, trainN, maxLen, V, PAD, BOS, EOS);
        ToySeq2SeqDataset validData = ToySeq2SeqDataset.make(ToySeq2SeqDataset.Task.COPY, validN, maxLen, V, PAD, BOS, EOS);

        Trainer trainer = new Trainer(model, opt, ceLoss, V, PAD);


        // Epoch loops

        Random r = new Random();

        for (int e = 1; e <= epochs; e++) {
            double[] trainRes = trainer.trainEpoch(trainData, batchSize);
            System.out.printf("epoch %d | loss=%.4f | tokenAcc=%.4f%n", e, trainRes[0], trainRes[1]);

            // For validataion


            int idx = r.nextInt(validN);
            Tensor valSrcId = validData.batchSrc(idx, 1);
            Tensor generated = Generator.greedyDecode(model, valSrcId, maxLen, BOS, EOS, PAD);

            System.out.println(valSrcId);
            System.out.println(generated);


            if(trainRes[1] >= 0.95) break;
        }

        // Validation

        for(int i=0;i<validN;i++) {
            Tensor validSrcIn = validData.batchSrc(i, 1);
            Tensor generated = Generator.greedyDecode(model, validSrcIn, maxLen, BOS, EOS, PAD);
            System.out.println(validSrcIn);
            System.out.println(generated);
        }
        */
    }
}
