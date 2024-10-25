package model;

import persistence.Writable;

// Represents a layer in a neural network.
public abstract class Layer implements Writable {

    // MODIFIES: this
    // EFFECTS: processes the input tensor and returns the output tensor;
    // throws IllegalArgumentException if input is null
    public abstract Tensor forward(Tensor input);

    // EFFECTS: computes and returns the gradient tensor for the previous layer;
    // throws IllegalArgumentException if gradient is null
    public abstract Tensor backward(Tensor gradient);

    // MODIFIES: this
    // EFFECTS: updates the layer's parameters using the optimizer;
    // throws IllegalArgumentException if optimizer is null
    public abstract void updateParameters(Optimizer optimizer);
}
