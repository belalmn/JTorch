package model;

// Represents an activation layer in a neural network.
public class ActivationLayer extends Layer {

    private String activationFunction;

    // EFFECTS: initializes the activation layer with the specified function;
    // throws IllegalArgumentException if activationFunction is null or unsupported
    public ActivationLayer(String activationFunction) {
        // TODO: implement constructor
    }

    // MODIFIES: this
    // EFFECTS: applies activation function to input tensor;
    // throws IllegalArgumentException if input is null
    public Tensor forward(Tensor input) {
        return null; // stub
    }

    // EFFECTS: computes gradient of activation function and multiplies element-wise;
    // throws IllegalArgumentException if gradient is null
    public Tensor backward(Tensor gradient) {
        return null; // stub
    }

    // EFFECTS: does nothing as activation layers typically have no parameters;
    // throws IllegalArgumentException if optimizer is null
    public void updateParameters(Optimizer optimizer) {
        // TODO: implement method
    }
}
