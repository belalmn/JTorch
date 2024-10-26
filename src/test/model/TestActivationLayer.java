package model;

import static org.junit.jupiter.api.Assertions.*;

import org.json.JSONObject;
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
        activationLayer.forward(inputTensor);
        Tensor inputGradient = activationLayer.backward(gradientTensor);
        assertNotNull(inputGradient);
    }

    @Test
    void testBackwardWithNullGradient() {
        assertThrows(IllegalArgumentException.class, () -> {
            activationLayer.backward(null);
        });
    }

    @Test
    void testUpdateParameters() {
        Optimizer optimizer = new SgdOptimizer(0.01);
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

    @Test
    void testToJson() {
        JSONObject json = activationLayer.toJson();
        assertNotNull(json);
        assertEquals("ActivationLayer", json.getString("type"));
        assertEquals("relu", json.getString("activationFunction"));
    }

    @Test
    void testFromJson() {
        JSONObject json = activationLayer.toJson();
        ActivationLayer deserializedLayer = ActivationLayer.fromJson(json);
        assertNotNull(deserializedLayer);
        assertEquals(
            activationLayer.getActivationFunction(),
            deserializedLayer.getActivationFunction()
        );
    }

    @Test
    void testSerializationRoundTrip() {
        JSONObject json = activationLayer.toJson();
        ActivationLayer deserializedLayer = ActivationLayer.fromJson(json);
        assertNotNull(deserializedLayer);
        assertEquals(
            activationLayer.getActivationFunction(),
            deserializedLayer.getActivationFunction()
        );
    }

    @Test
    void testForwardSigmoid() {
        activationLayer = new ActivationLayer("sigmoid");
        double[][] inputData = {{-1.0, 0.0, 1.0}};
        inputTensor = new Tensor(inputData);

        outputTensor = activationLayer.forward(inputTensor);
        assertNotNull(outputTensor);

        double[][] expectedData = new double[1][3];
        for (int i = 0; i < 3; i++) {
            double x = inputData[0][i];
            expectedData[0][i] = 1 / (1 + Math.exp(-x));
        }

        assertArrayEquals(expectedData[0], outputTensor.getData()[0], 0.0001);
    }

    @Test
    void testBackwardSigmoid() {
        activationLayer = new ActivationLayer("sigmoid");
        double[][] inputData = {{-1.0, 0.0, 1.0}};
        inputTensor = new Tensor(inputData);
        activationLayer.forward(inputTensor); // Forward pass to set inputData in activationLayer

        double[][] gradientData = {{0.1, 0.2, 0.3}};
        Tensor gradientTensor = new Tensor(gradientData);

        Tensor inputGradient = activationLayer.backward(gradientTensor);
        assertNotNull(inputGradient);

        double[][] expectedInputGradient = new double[1][3];
        for (int i = 0; i < 3; i++) {
            double x = inputData[0][i];
            double sigmoid = 1 / (1 + Math.exp(-x));
            double derivative = sigmoid * (1 - sigmoid);
            expectedInputGradient[0][i] = gradientData[0][i] * derivative;
        }

        assertArrayEquals(expectedInputGradient[0], inputGradient.getData()[0], 0.0001);
    }

    @Test
    void testConstructorWithSigmoid() {
        activationLayer = new ActivationLayer("sigmoid");
        assertEquals("sigmoid", activationLayer.getActivationFunction());
    }
}
