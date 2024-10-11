package model;

import static org.junit.jupiter.api.Assertions.*;
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
        Optimizer optimizer = new SGDOptimizer(0.01);

        assertDoesNotThrow(() -> {
            network.train(inputs, targets, 10, optimizer);
        });
    }

    @Test
    void testTrainWithNullInputs() {
        List<Tensor> targets = new ArrayList<>();
        Optimizer optimizer = new SGDOptimizer(0.01);
        assertThrows(IllegalArgumentException.class, () -> {
            network.train(null, targets, 10, optimizer);
        });
    }

    @Test
    void testTrainWithNullTargets() {
        List<Tensor> inputs = new ArrayList<>();
        Optimizer optimizer = new SGDOptimizer(0.01);
        assertThrows(IllegalArgumentException.class, () -> {
            network.train(inputs, null, 10, optimizer);
        });
    }

    @Test
    void testTrainWithMismatchedSizes() {
        List<Tensor> inputs = new ArrayList<>();
        List<Tensor> targets = new ArrayList<>();
        inputs.add(inputTensor);
        Optimizer optimizer = new SGDOptimizer(0.01);
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
        Optimizer optimizer = new SGDOptimizer(0.01);
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
}
