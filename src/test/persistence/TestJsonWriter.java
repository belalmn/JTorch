package persistence;

import model.*;
import org.json.JSONObject;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import java.io.IOException;

public class TestJsonWriter {
    private Tensor tensor;
    private static final String TENSOR_FILE = "./data/testTensorWrite.json";

    @BeforeEach
    void runBefore() {
        double[][] data = {{1.0, 2.0}, {3.0, 4.0}};
        tensor = new Tensor(data);
    }

    @Test
    public void testWriteTensor() {
        JsonWriter writer = new JsonWriter(TENSOR_FILE);
        try {
            writer.open();
            writer.write(tensor);
            writer.close();

            JsonReader reader = new JsonReader(TENSOR_FILE);
            JSONObject json = reader.readJson();
            Tensor loadedTensor = Tensor.fromJson(json);
            assert2dArrayEquals(tensor.getData(), loadedTensor.getData(), 0.0001);
        } catch (IOException e) {
            fail("IOException should not have occurred");
        }
    }

    private static void assert2dArrayEquals(double[][] expected, double[][] actual, double delta) {
        assertEquals(expected.length, actual.length, "Row count mismatch");
        for (int i = 0; i < expected.length; i++) {
            assertArrayEquals(expected[i], actual[i], delta, "Mismatch at row " + i);
        }
    }
}
