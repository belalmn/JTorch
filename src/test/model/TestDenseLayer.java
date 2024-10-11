package model;

import static org.junit.jupiter.api.Assertions.*;
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
        // TODO: More assertions
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
        Tensor inputGradient = denseLayer.backward(gradientTensor);
        assertNotNull(inputGradient);
        // TODO: More assertions
    }

    @Test
    void testBackwardWithNullGradient() {
        assertThrows(IllegalArgumentException.class, () -> {
            denseLayer.backward(null);
        });
    }

    @Test
    void testUpdateParameters() {
        Optimizer optimizer = new SGDOptimizer(0.01);
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
}
