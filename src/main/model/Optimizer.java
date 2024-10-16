package model;

// Represents an optimizer used for updating neural network parameters during training.
public abstract class Optimizer {

    // MODIFIES: layer
    // EFFECTS: updates the parameters of the layer;
    // throws IllegalArgumentException if layer is null
    public abstract void updateParameters(Layer layer);
}
