package tensor;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;

public class Nd4jInit {
    public static void configure() {
        Nd4j.setDefaultDataTypes(DataType.FLOAT, DataType.FLOAT);
    }
}
