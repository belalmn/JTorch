package model;

import static org.junit.jupiter.api.Assertions.*;

import org.json.JSONObject;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.ArrayList;

public class TestNeuralNetwork {
    private NeuralNetwork network;
    private Tensor inputTensor;
    private Tensor outputTensor;

    @BeforeEach
    void runBefore() {
        network = new NeuralNetwork();
        network.addLayer(new DenseLayer(2, 3));
        network.addLayer(new ActivationLayer("relu"));
        network.addLayer(new DenseLayer(3, 1));
        double[][] inputData = {{1.0, 2.0}};
        inputTensor = new Tensor(inputData);
    }

    @Test
    void testConstructor() {
        assertNotNull(network);
    }

    @Test
    void testAddLayer() {
        NeuralNetwork newNetwork = new NeuralNetwork();
        Layer layer = new DenseLayer(2, 3);
        newNetwork.addLayer(layer);
        assertEquals(1, newNetwork.getLayers().size());
    }

    @Test
    void testAddLayerWithNullLayer() {
        NeuralNetwork newNetwork = new NeuralNetwork();
        assertThrows(IllegalArgumentException.class, () -> {
            newNetwork.addLayer(null);
        });
    }

    @Test
    void testPredict() {
        outputTensor = network.predict(inputTensor);
        assertNotNull(outputTensor);
    }

    @Test
    void testPredictWithNullInput() {
        assertThrows(IllegalArgumentException.class, () -> {
            network.predict(null);
        });
    }

    @Test
    void testTrain() {
        List<Tensor> inputs = new ArrayList<>();
        List<Tensor> targets = new ArrayList<>();
        inputs.add(inputTensor);
        double[][] targetData = {{1.0}};
        targets.add(new Tensor(targetData));
        Optimizer optimizer = new SgdOptimizer(0.01);

        assertDoesNotThrow(() -> {
            network.train(inputs, targets, 10, optimizer);
        });
    }

    @Test
    void testTrainWithNullInputs() {
        List<Tensor> targets = new ArrayList<>();
        Optimizer optimizer = new SgdOptimizer(0.01);
        assertThrows(IllegalArgumentException.class, () -> {
            network.train(null, targets, 10, optimizer);
        });
    }

    @Test
    void testTrainWithNullTargets() {
        List<Tensor> inputs = new ArrayList<>();
        Optimizer optimizer = new SgdOptimizer(0.01);
        assertThrows(IllegalArgumentException.class, () -> {
            network.train(inputs, null, 10, optimizer);
        });
    }

    @Test
    void testTrainWithMismatchedSizes() {
        List<Tensor> inputs = new ArrayList<>();
        List<Tensor> targets = new ArrayList<>();
        inputs.add(inputTensor);
        Optimizer optimizer = new SgdOptimizer(0.01);
        assertThrows(IllegalArgumentException.class, () -> {
            network.train(inputs, targets, 10, optimizer);
        });
    }

    @Test
    void testTrainWithNonPositiveEpochs() {
        List<Tensor> inputs = new ArrayList<>();
        List<Tensor> targets = new ArrayList<>();
        inputs.add(inputTensor);
        targets.add(new Tensor(new double[][]{{1.0}}));
        Optimizer optimizer = new SgdOptimizer(0.01);
        assertThrows(IllegalArgumentException.class, () -> {
            network.train(inputs, targets, 0, optimizer);
        });
    }

    @Test
    void testTrainWithNullOptimizer() {
        List<Tensor> inputs = new ArrayList<>();
        List<Tensor> targets = new ArrayList<>();
        inputs.add(inputTensor);
        targets.add(new Tensor(new double[][]{{1.0}}));
        assertThrows(IllegalArgumentException.class, () -> {
            network.train(inputs, targets, 10, null);
        });
    }

    @Test
    void testGetArchitecture() {
        String architecture = network.getArchitecture();
        assertNotNull(architecture);
        assertTrue(architecture.contains("DenseLayer"));
        assertTrue(architecture.contains("ActivationLayer"));
    }

    @Test
    public void testToJson() {
        JSONObject json = network.toJson();
        assertNotNull(json);
        assertTrue(json.has("layers"));
        assertEquals(3, json.getJSONArray("layers").length());
    }

    @Test
    public void testFromJson() {
        JSONObject json = network.toJson();
        NeuralNetwork deserializedNetwork = NeuralNetwork.fromJson(json);
        assertNotNull(deserializedNetwork);

        List<Layer> originalLayers = network.getLayers();
        List<Layer> deserializedLayers = deserializedNetwork.getLayers();
        assertEquals(originalLayers.size(), deserializedLayers.size());

        // Compare each layer
        for (int i = 0; i < originalLayers.size(); i++) {
            Layer originalLayer = originalLayers.get(i);
            Layer deserializedLayer = deserializedLayers.get(i);
            assertEquals(
                originalLayer.getClass(),
                deserializedLayer.getClass()
            );
            if (originalLayer instanceof DenseLayer) {
                DenseLayer originalDense = (DenseLayer) originalLayer;
                DenseLayer deserializedDense = (DenseLayer) deserializedLayer;
                assert2dArrayEquals(
                    originalDense.getWeights().getData(),
                    deserializedDense.getWeights().getData(),
                    0.0001
                );
                assert2dArrayEquals(
                    originalDense.getBiases().getData(),
                    deserializedDense.getBiases().getData(),
                    0.0001
                );
            } else if (originalLayer instanceof ActivationLayer) {
                ActivationLayer originalActivation = (ActivationLayer) originalLayer;
                ActivationLayer deserializedActivation = (ActivationLayer) deserializedLayer;
                assertEquals(
                    originalActivation.getActivationFunction(),
                    deserializedActivation.getActivationFunction()
                );
            }
        }
    }

    @Test
    public void testSerializationRoundTrip() {
        JSONObject json = network.toJson();
        NeuralNetwork deserializedNetwork = NeuralNetwork.fromJson(json);
        assertNotNull(deserializedNetwork);

        List<Layer> originalLayers = network.getLayers();
        List<Layer> deserializedLayers = deserializedNetwork.getLayers();
        assertEquals(originalLayers.size(), deserializedLayers.size());

        // Comparing each layer
        for (int i = 0; i < originalLayers.size(); i++) {
            Layer originalLayer = originalLayers.get(i);
            Layer deserializedLayer = deserializedLayers.get(i);
            assertEquals(
                originalLayer.getClass(),
                deserializedLayer.getClass()
            );
            if (originalLayer instanceof DenseLayer) {
                DenseLayer originalDense = (DenseLayer) originalLayer;
                DenseLayer deserializedDense = (DenseLayer) deserializedLayer;
                assert2dArrayEquals(
                    originalDense.getWeights().getData(),
                    deserializedDense.getWeights().getData(),
                    0.0001
                );
                assert2dArrayEquals(
                    originalDense.getBiases().getData(),
                    deserializedDense.getBiases().getData(),
                    0.0001
                );
            } else if (originalLayer instanceof ActivationLayer) {
                ActivationLayer originalActivation = (ActivationLayer) originalLayer;
                ActivationLayer deserializedActivation = (ActivationLayer) deserializedLayer;
                assertEquals(
                    originalActivation.getActivationFunction(),
                    deserializedActivation.getActivationFunction()
                );
            }
        }
    }

    // Helper method
    private static void assert2dArrayEquals(double[][] expected, double[][] actual, double delta) {
        assertEquals(expected.length, actual.length, "Row count mismatch");
        for (int i = 0; i < expected.length; i++) {
            assertArrayEquals(expected[i], actual[i], delta, "Mismatch at row " + i);
        }
    }
}
