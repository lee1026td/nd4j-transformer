package nn.loss;

import tensor.Tensor;

public class CrossEntropyLoss implements Loss {

    private Tensor prob;        // [B, T, V]
    private Tensor targets;     // [B, T] (int)
    private Tensor mask;        // [B, T]
    private double denom;       // sum(mask)

    @Override
    public double forward(Tensor preds, Tensor tgts) {
        return forward(preds, tgts, null);
    }

    @Override
    public double forward(Tensor logits, Tensor targets, Tensor mask) {
        int B = logits.size(0), T = logits.size(1), V = logits.size(2);
        if(targets.size(0) != B || targets.size(1) != T)
            throw new IllegalArgumentException("Targets shape mismatch");
        if(mask.size(0) != B || mask.size(1) != T)
            throw new IllegalArgumentException("Mask shape mismatch");

        // Caching
        this.targets = targets;
        this.mask = mask;

        // Log-softmax
        Tensor maxLog = logits.max(-1, true);       // [B, T, 1]
        Tensor z = logits.sub(maxLog);                           // [B, T, V]
        Tensor expZ = z.exp();
        Tensor sumExp = expZ.sum(-1, true);         // [B, T, 1]

        this.prob = expZ.div(sumExp);
        Tensor logP = z.sub(sumExp.log());

        // NLL = -logP(target)
        double num = 0.0;
        double denom = 0.0;

        for(int i=0;i<B;i++) {
            for(int j=0;j<T;j++) {
                int y = targets.getInt(i, j);
                double m = mask.getDouble(i, j);

                if(m == 0.0) continue;      // PAD skipping

                double lp = logP.getDouble(i, j, y);
                num += (-lp) * m;
                denom += m;
            }
        }

        this.denom = (denom > 0.0) ? denom : 1e-12;

        double loss = num / this.denom;

        return loss;
    }

    @Override
    public Tensor backward() {
        int B = prob.size(0), T = prob.size(1), V = prob.size(2);

        Tensor dLogits = prob.dup();        // [B, T, V]
        // sub 1 only target pos
        for(int i=0;i<B;i++) {
            for(int j=0;j<T;j++) {
                int y = targets.getInt(i, j);
                double v = dLogits.getDouble(i, j, y) - 1.0;

                dLogits.set(v, i, j, y);        // put v in (i, j, y)
            }
        }

        Tensor scale = mask.div(denom).reshape(B, T, 1);        // [B, T] -> [B, T, 1]
        return dLogits.mul(scale);                              // [B, T, V]
    }
}
