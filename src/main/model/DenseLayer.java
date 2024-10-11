package model;

// Represents a fully connected (dense) layer in a neural network.
public class DenseLayer extends Layer {

    private Tensor weights;
    private Tensor biases;

    // EFFECTS: initializes weights and biases randomly;
    // throws IllegalArgumentException if inputSize <= 0 or outputSize <= 0
    public DenseLayer(int inputSize, int outputSize) {
        // TODO: implement constructor
    }

    // MODIFIES: this
    // EFFECTS: computes output = input * weights + biases;
    // throws IllegalArgumentException if input is null or dimensions are invalid
    public Tensor forward(Tensor input) {
        return null; // stub
    }

    // MODIFIES: this
    // EFFECTS: computes gradients with respect to weights, biases, and input;
    // throws IllegalArgumentException if gradient is null
    public Tensor backward(Tensor gradient) {
        return null; // stub
    }

    // MODIFIES: this
    // EFFECTS: updates weights and biases using computed gradients;
    // throws IllegalArgumentException if optimizer is null
    public void updateParameters(Optimizer optimizer) {
        // TODO: implement method
    }
}
