package model;

import static org.junit.jupiter.api.Assertions.*;

import org.json.JSONObject;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

public class TestSgdOptimizer {
    private SgdOptimizer optimizer;
    private DenseLayer layer;
    private Tensor inputTensor;
    private Tensor gradientTensor;

    @BeforeEach
    void runBefore() {
        optimizer = new SgdOptimizer(0.01);
        layer = new DenseLayer(2, 3);

        // Prepare input tensor
        double[][] inputData = {{1.0, 2.0}};
        inputTensor = new Tensor(inputData);

        // Prepare gradient tensor (matching output size)
        double[][] gradientData = {{0.1, 0.2, 0.3}};
        gradientTensor = new Tensor(gradientData);
    }

    @Test
    void testConstructor() {
        assertEquals(0.01, optimizer.getLearningRate());
    }

    @Test
    void testConstructorWithInvalidLearningRate() {
        assertThrows(IllegalArgumentException.class, () -> {
            new SgdOptimizer(0);
        });
        assertThrows(IllegalArgumentException.class, () -> {
            new SgdOptimizer(-0.01);
        });
    }

    @Test
    void testUpdateParameters() {
        // Perform forward pass
        layer.forward(inputTensor);

        // Perform backward pass to compute gradients
        layer.backward(gradientTensor);

        // Store original weights and biases
        Tensor originalWeights = layer.getWeights();
        Tensor originalBiases = layer.getBiases();

        // Update parameters
        assertDoesNotThrow(() -> {
            optimizer.updateParameters(layer);
        });

        // Get updated weights and biases
        Tensor updatedWeights = layer.getWeights();
        Tensor updatedBiases = layer.getBiases();

        // Assertions to check if weights and biases have been updated
        assertNotNull(updatedWeights);
        assertNotNull(updatedBiases);
        assertFalse(arraysEqual(originalWeights.getData(), updatedWeights.getData()));
        assertFalse(arraysEqual(originalBiases.getData(), updatedBiases.getData()));
    }

    @Test
    void testUpdateParametersWithNullLayer() {
        assertThrows(IllegalArgumentException.class, () -> {
            optimizer.updateParameters(null);
        });
    }

    @Test
    public void testToJson() {
        JSONObject json = optimizer.toJson();
        assertNotNull(json);
        assertEquals("SgdOptimizer", json.getString("type"));
        assertEquals(0.01, json.getDouble("learningRate"), 0.0001);
    }

    @Test
    public void testFromJson() {
        JSONObject json = optimizer.toJson();
        SgdOptimizer deserializedOptimizer = SgdOptimizer.fromJson(json);
        assertNotNull(deserializedOptimizer);
        assertEquals(
            optimizer.getLearningRate(),
            deserializedOptimizer.getLearningRate(),
            0.0001
        );
    }

    @Test
    public void testSerializationRoundTrip() {
        JSONObject json = optimizer.toJson();
        SgdOptimizer deserializedOptimizer = SgdOptimizer.fromJson(json);
        assertNotNull(deserializedOptimizer);
        assertEquals(
            optimizer.getLearningRate(),
            deserializedOptimizer.getLearningRate(),
            0.0001
        );
    }

    // Helper method to compare two 2D arrays
    private boolean arraysEqual(double[][] a, double[][] b) {
        if (a.length != b.length) {
            return false;
        }
        for (int i = 0; i < a.length; i++) {
            if (a[i].length != b[i].length) {
                return false;
            }
            for (int j = 0; j < a[i].length; j++) {
                if (a[i][j] != b[i][j]) {
                    return false;
                }
            }
        }
        return true;
    }

    @Test
    void testSetLearningRateValid() {
        optimizer.setLearningRate(0.05);
        assertEquals(0.05, optimizer.getLearningRate(), 0.0001);
    }

    @Test
    void testSetLearningRateInvalid() {
        assertThrows(IllegalArgumentException.class, () -> {
            optimizer.setLearningRate(0);
        });

        assertThrows(IllegalArgumentException.class, () -> {
            optimizer.setLearningRate(-0.01);
        });
    }
}
