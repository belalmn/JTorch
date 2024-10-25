package model;

import static org.junit.jupiter.api.Assertions.*;

import org.json.JSONObject;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

public class TestTensor {
    private Tensor tensor1;
    private Tensor tensor2;
    private Tensor tensor3;
    private double[][] data1;
    private double[][] data2;
    private double[][] data3;
    
    @BeforeEach
    void runBefore() {
        data1 = new double[][]{{1.0, 2.0}, {3.0, 4.0}};
        data2 = new double[][]{{5.0, 6.0}, {7.0, 8.0}};
        data3 = new double[][]{{1.0, 2.0, 3.0}};
        tensor1 = new Tensor(data1);
        tensor2 = new Tensor(data2);
    }

    @Test
    void testConstructor() {
        assertArrayEquals(data1, tensor1.getData());
    }

    @Test
    void testConstructorWithNullData() {
        assertThrows(IllegalArgumentException.class, () -> {
            new Tensor(null);
        });
    }

    @Test
    void testAdd() {
        tensor1.add(tensor2);
        double[][] expectedData = {{6.0, 8.0}, {10.0, 12.0}};
        assertArrayEquals(expectedData, tensor1.getData());
    }

    @Test
    void testAddWithNullOther() {
        assertThrows(IllegalArgumentException.class, () -> {
            tensor1.add(null);
        });
    }

    @Test
    void testAddWithMismatchedDimensions() {
        tensor3 = new Tensor(data3);
        assertThrows(IllegalArgumentException.class, () -> {
            tensor1.add(tensor3);
        });
    }

    @Test
    void testMultiply() {
        tensor1.multiply(tensor2);
        double[][] expectedData = {{5.0, 12.0}, {21.0, 32.0}};
        assertArrayEquals(expectedData, tensor1.getData());
    }

    @Test
    void testMultiplyWithNullOther() {
        assertThrows(IllegalArgumentException.class, () -> {
            tensor1.multiply(null);
        });
    }

    @Test
    void testMultiplyWithMismatchedDimensions() {
        tensor3 = new Tensor(data3);
        assertThrows(IllegalArgumentException.class, () -> {
            tensor1.multiply(tensor3);
        });
    }

    @Test
    public void testToJson() {
        JSONObject json = tensor1.toJson();
        assertNotNull(json);
        assertTrue(json.has("data"));
    }

    @Test
    public void testFromJson() {
        JSONObject json = tensor1.toJson();
        Tensor deserializedTensor = Tensor.fromJson(json);
        assertNotNull(deserializedTensor);
        assert2dArrayEquals(tensor1.getData(), deserializedTensor.getData(), 0.0001);
    }

    @Test
    public void testSerializationRoundTrip() {
        JSONObject json = tensor1.toJson();
        Tensor deserializedTensor = Tensor.fromJson(json);
        assertNotNull(deserializedTensor);
        assert2dArrayEquals(tensor1.getData(), deserializedTensor.getData(), 0.0001);
    }

    // Helper method to compare 2D arrays
    private static void assert2dArrayEquals(double[][] expected, double[][] actual, double delta) {
        assertEquals(expected.length, actual.length, "Row count mismatch");
        for (int i = 0; i < expected.length; i++) {
            assertArrayEquals(expected[i], actual[i], delta, "Mismatch at row " + i);
        }
    }
}
