package model;

// Represents metrics for evaluating the performance of the neural network.
public class Metric {

    // EFFECTS: computes and returns the loss value (e.g., Mean Squared Error);
    // throws IllegalArgumentException if output or target is null or dimensions do not match
    public double calculateLoss(Tensor output, Tensor target) {
        return 0.0; // stub
    }

    // EFFECTS: computes and returns the accuracy;
    // throws IllegalArgumentException if output or target is null or dimensions do not match
    public double calculateAccuracy(Tensor output, Tensor target) {
        return 0.0; // stub
    }

    // EFFECTS: computes the gradient of the loss function with respect to the output;
    // throws IllegalArgumentException if output or target is null or dimensions do not match
    public Tensor lossGradient(Tensor output, Tensor target) {
        return null; // stub
    }
}
