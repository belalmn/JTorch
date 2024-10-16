package model;

import java.util.List;
import java.util.ArrayList;

// Represents a neural network composed of multiple layers.
public class NeuralNetwork {

    private List<Layer> layers;
    private TrainingListener trainingListener;


    // EFFECTS: initializes an empty list of layers
    public NeuralNetwork() {
        layers = new ArrayList<>();
    }

    // MODIFIES: this
    // EFFECTS: adds the layer to the network;
    // throws IllegalArgumentException if layer is null
    public void addLayer(Layer layer) {
        if (layer == null) {
            throw new IllegalArgumentException("Layer cannot be null");
        }
        layers.add(layer);
    }

    // MODIFIES: this
    // EFFECTS: sets the training listener
    public void setTrainingListener(TrainingListener listener) {
        this.trainingListener = listener;
    }

    // MODIFIES: this
    // EFFECTS: trains the network on the data for the specified number of epochs;
    // throws IllegalArgumentException if inputs or targets are null,
    // sizes do not match, epochs <= 0, or optimizer is null;
    // notifies UI of loss for each epoch throughout training
    public void train(List<Tensor> inputs, List<Tensor> targets, int epochs, Optimizer optimizer) {
        if (inputs == null || targets == null || optimizer == null || epochs <= 0 || inputs.size() != targets.size()) {
            throw new IllegalArgumentException("Invalid training parameters");
        }
        Metric metric = new Metric();
        for (int epoch = 0; epoch < epochs; epoch++) {
            trainEpoch(inputs, targets, optimizer, metric);
            double totalLoss = trainEpoch(inputs, targets, optimizer, metric);
            double averageLoss = totalLoss / inputs.size();

            // Notifies listener of new epoch and loss
            if (trainingListener != null) {
                trainingListener.onEpochEnd(epoch + 1, epochs, averageLoss);
            }
        }
    }

    // Helper method to train for one epoch
    private double trainEpoch(List<Tensor> inputs, List<Tensor> targets, Optimizer optimizer, Metric metric) {
        double totalLoss = 0;
        for (int i = 0; i < inputs.size(); i++) {
            Tensor output = forwardPass(inputs.get(i));
            double loss = metric.calculateLoss(output, targets.get(i));
            totalLoss += loss;
            Tensor lossGradient = metric.lossGradient(output, targets.get(i));
            backwardPass(lossGradient);
            updateParameters(optimizer);
        }
        return totalLoss;
    }

    // Helper method for the forward pass
    private Tensor forwardPass(Tensor input) {
        Tensor output = input;
        for (Layer layer : layers) {
            output = layer.forward(output);
        }
        return output;
    }

    // Helper method for the backward pass
    private void backwardPass(Tensor lossGradient) {
        Tensor grad = lossGradient;
        for (int j = layers.size() - 1; j >= 0; j--) {
            grad = layers.get(j).backward(grad);
        }
    }

    // Helper method to update parameters of all layers
    private void updateParameters(Optimizer optimizer) {
        for (Layer layer : layers) {
            layer.updateParameters(optimizer);
        }
    }

    // EFFECTS: computes the output of the network for the given input;
    // throws IllegalArgumentException if input is null
    public Tensor predict(Tensor input) {
        if (input == null) {
            throw new IllegalArgumentException("Input cannot be null");
        }
        return forwardPass(input);
    }

    // EFFECTS: returns a string listing the layers and their configurations
    public String getArchitecture() {
        StringBuilder sb = new StringBuilder();
        for (Layer layer : layers) {
            sb.append(layer.getClass().getSimpleName());
            if (layer instanceof DenseLayer) {
                DenseLayer denseLayer = (DenseLayer) layer;
                sb.append(" (input size: ").append(denseLayer.getWeights().getData().length);
                sb.append(", output size: ").append(denseLayer.getWeights().getData()[0].length).append(")");
            } else if (layer instanceof ActivationLayer) {
                ActivationLayer activationLayer = (ActivationLayer) layer;
                sb.append(" (activation: ").append(activationLayer.getActivationFunction()).append(")");
            }
            sb.append("\n");
        }
        return sb.toString();
    }

    // Getter for the list of layers (added for testing purposes)
    public List<Layer> getLayers() {
        return layers;
    }
}
