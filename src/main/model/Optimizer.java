package model;

import persistence.Writable;

// Represents an optimizer used for updating neural network parameters during training.
public abstract class Optimizer implements Writable {

    // MODIFIES: layer
    // EFFECTS: updates the parameters of the layer;
    // throws IllegalArgumentException if layer is null
    public abstract void updateParameters(Layer layer);
}
