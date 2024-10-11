package model;

import java.util.List;

// Represents a neural network composed of multiple layers.
public class NeuralNetwork {

    // EFFECTS: initializes an empty list of layers
    public NeuralNetwork() {
        // TODO: implement constructor
    }

    // MODIFIES: this
    // EFFECTS: adds the layer to the network;
    // throws IllegalArgumentException if layer is null
    public void addLayer(Layer layer) {
        // TODO: implement method
    }

    // MODIFIES: this
    // EFFECTS: trains the network on the data for the specified number of epochs;
    // throws IllegalArgumentException if inputs or targets are null,
    // sizes do not match, epochs <= 0, or optimizer is null
    public void train(List<Tensor> inputs, List<Tensor> targets, int epochs, Optimizer optimizer) {
        // TODO: implement method
    }

    // EFFECTS: computes the output of the network for the given input;
    // throws IllegalArgumentException if input is null
    public Tensor predict(Tensor input) {
        return null; // stub
    }

    // EFFECTS: returns a string listing the layers and their configurations
    public String getArchitecture() {
        return ""; // stub
    }

    public List<Layer> getLayers() {
        return null; // stub
    }
}
