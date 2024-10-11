package model;

// Represents an optimizer used for updating neural network parameters during training.
public interface Optimizer {

    // MODIFIES: layer
    // EFFECTS: updates the parameters of the layer;
    // throws IllegalArgumentException if layer is null
    void updateParameters(Layer layer);
}
