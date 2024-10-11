package model;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

public class TestActivationLayer {
    private ActivationLayer activationLayer;
    private Tensor inputTensor;
    private Tensor outputTensor;

    @BeforeEach
    void runBefore() {
        activationLayer = new ActivationLayer("relu");
        double[][] inputData = {{-1.0, 0.0, 1.0}};
        inputTensor = new Tensor(inputData);
    }

    @Test
    void testConstructor() {
        assertNotNull(activationLayer);
    }

    @Test
    void testConstructorWithNullActivationFunction() {
        assertThrows(IllegalArgumentException.class, () -> {
            new ActivationLayer(null);
        });
    }

    @Test
    void testConstructorWithUnsupportedActivationFunction() {
        assertThrows(IllegalArgumentException.class, () -> {
            new ActivationLayer("unsupported");
        });
    }

    @Test
    void testForward() {
        outputTensor = activationLayer.forward(inputTensor);
        assertNotNull(outputTensor);
        double[][] expectedData = {{0.0, 0.0, 1.0}};
        assertArrayEquals(expectedData, outputTensor.getData());
    }

    @Test
    void testForwardWithNullInput() {
        assertThrows(IllegalArgumentException.class, () -> {
            activationLayer.forward(null);
        });
    }

    @Test
    void testBackward() {
        double[][] gradientData = {{0.1, 0.2, 0.3}};
        Tensor gradientTensor = new Tensor(gradientData);
        Tensor inputGradient = activationLayer.backward(gradientTensor);
        assertNotNull(inputGradient);
        // TODO: More assertions
    }

    @Test
    void testBackwardWithNullGradient() {
        assertThrows(IllegalArgumentException.class, () -> {
            activationLayer.backward(null);
        });
    }

    @Test
    void testUpdateParameters() {
        Optimizer optimizer = new SGDOptimizer(0.01);
        assertDoesNotThrow(() -> {
            activationLayer.updateParameters(optimizer);
        });
    }

    @Test
    void testUpdateParametersWithNullOptimizer() {
        assertThrows(IllegalArgumentException.class, () -> {
            activationLayer.updateParameters(null);
        });
    }
}
