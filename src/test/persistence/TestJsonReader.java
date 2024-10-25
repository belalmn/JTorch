package persistence;

import model.*;
import org.json.JSONObject;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import java.io.IOException;

public class TestJsonReader {
    private static final String TENSOR_FILE = "./data/testTensorRead.json";

    @Test
    public void testReadTensor() {
        JsonReader reader = new JsonReader(TENSOR_FILE);
        try {
            JSONObject json = reader.readJson();
            Tensor tensor = Tensor.fromJson(json);
            
            double[][] expectedData = {{1.0, 2.0}, {3.0, 4.0}};
            assert2dArrayEquals(expectedData, tensor.getData(), 0.0001);
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
