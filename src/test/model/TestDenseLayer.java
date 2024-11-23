package model;

import static org.junit.jupiter.api.Assertions.*;

import org.json.JSONObject;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

public class TestDenseLayer {
    private DenseLayer denseLayer;
    private Tensor inputTensor;
    private Tensor outputTensor;
    private int inputSize = 2;
    private int outputSize = 3;

    @BeforeEach
    void runBefore() {
        denseLayer = new DenseLayer(inputSize, outputSize);
        double[][] inputData = {{1.0, 2.0}};
        inputTensor = new Tensor(inputData);
    }

    @Test
    void testConstructor() {
        assertNotNull(denseLayer);
    }

    @Test
    void testConstructorWithInvalidSizes() {
        assertThrows(IllegalArgumentException.class, () -> {
            new DenseLayer(0, outputSize);
        });
        assertThrows(IllegalArgumentException.class, () -> {
            new DenseLayer(inputSize, -1);
        });
    }

    @Test
    void testForward() {
        outputTensor = denseLayer.forward(inputTensor);
        assertNotNull(outputTensor);
    }

    @Test
    void testForwardWithNullInput() {
        assertThrows(IllegalArgumentException.class, () -> {
            denseLayer.forward(null);
        });
    }

    @Test
    void testForwardWithInvalidDimensions() {
        Tensor invalidInput = new Tensor(new double[][]{{1.0}});
        assertThrows(IllegalArgumentException.class, () -> {
            denseLayer.forward(invalidInput);
        });
    }

    @Test
    void testBackward() {
        double[][] gradientData = {{0.1, 0.2, 0.3}};
        Tensor gradientTensor = new Tensor(gradientData);
        denseLayer.forward(inputTensor);
        Tensor inputGradient = denseLayer.backward(gradientTensor);
        assertNotNull(inputGradient);
    }

    @Test
    void testBackwardWithNullGradient() {
        assertThrows(IllegalArgumentException.class, () -> {
            denseLayer.backward(null);
        });
    }

    @Test
    void testUpdateParameters() {
        Optimizer optimizer = new SgdOptimizer(0.01);

        double[][] inputData = {{1.0, 2.0}};
        Tensor inputTensor = new Tensor(inputData);
        denseLayer.forward(inputTensor);
        double[][] gradientData = {{0.1, 0.2, 0.3}};
        Tensor gradientTensor = new Tensor(gradientData);
        denseLayer.backward(gradientTensor);

        assertDoesNotThrow(() -> {
            denseLayer.updateParameters(optimizer);
        });
    }

    @Test
    void testUpdateParametersWithNullOptimizer() {
        assertThrows(IllegalArgumentException.class, () -> {
            denseLayer.updateParameters(null);
        });
    }

    @Test
    public void testToJson() {
        JSONObject json = denseLayer.toJson();
        assertNotNull(json);
        assertEquals("DenseLayer", json.getString("type"));
        assertTrue(json.has("weights"));
        assertTrue(json.has("biases"));
    }

    @Test
    public void testFromJson() {
        JSONObject json = denseLayer.toJson();
        DenseLayer deserializedLayer = DenseLayer.fromJson(json);
        assertNotNull(deserializedLayer);

        assert2dArrayEquals(
            denseLayer.getWeights().getData(),
            deserializedLayer.getWeights().getData(),
            0.0001
        );
        assert2dArrayEquals(
            denseLayer.getBiases().getData(),
            deserializedLayer.getBiases().getData(),
            0.0001
        );
    }

    @Test
    public void testSerializationRoundTrip() {
        JSONObject json = denseLayer.toJson();
        DenseLayer deserializedLayer = DenseLayer.fromJson(json);
        assertNotNull(deserializedLayer);

        assert2dArrayEquals(
            denseLayer.getWeights().getData(),
            deserializedLayer.getWeights().getData(),
            0.0001
        );
        assert2dArrayEquals(
            denseLayer.getBiases().getData(),
            deserializedLayer.getBiases().getData(),
            0.0001
        );
    }

    @Test
    void testGetDescription() {
        String description = denseLayer.getDescription();
        assertNotNull(description);
        assertTrue(description.contains("Dense Layer (2 -> 3)"));
    }

    // Helper method
    private static void assert2dArrayEquals(double[][] expected, double[][] actual, double delta) {
        assertEquals(expected.length, actual.length, "Row count mismatch");
        for (int i = 0; i < expected.length; i++) {
            assertArrayEquals(expected[i], actual[i], delta, "Mismatch at row " + i);
        }
    }
}
