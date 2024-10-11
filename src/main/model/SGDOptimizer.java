package model;

// Implements the stochastic gradient descent (SGD) optimization algorithm.
public class SGDOptimizer implements Optimizer {

    private double learningRate;

    // EFFECTS: initializes the optimizer with the given learning rate;
    // throws IllegalArgumentException if learningRate <= 0
    public SGDOptimizer(double learningRate) {
        // TODO: implement constructor
    }

    // MODIFIES: layer
    // EFFECTS: updates the layer's parameters using SGD update rule;
    // throws IllegalArgumentException if layer is null
    public void updateParameters(Layer layer) {
        // TODO: implement method
    }

    public double getLearningRate(){
        return 0; // stub
    }
}
