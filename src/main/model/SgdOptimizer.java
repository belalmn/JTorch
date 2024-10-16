package model;

// Implements the stochastic gradient descent (SGD) optimization algorithm.
public class SgdOptimizer extends Optimizer {

    private double learningRate;

    // EFFECTS: initializes the optimizer with the given learning rate;
    // throws IllegalArgumentException if learningRate <= 0
    public SgdOptimizer(double learningRate) {
        if (learningRate <= 0) {
            throw new IllegalArgumentException("Learning rate must be positive");
        }
        this.learningRate = learningRate;
    }

    // MODIFIES: layer
    // EFFECTS: updates the layer's parameters using SGD update rule;
    // throws IllegalArgumentException if layer is null
    @Override
    public void updateParameters(Layer layer) {
        if (layer == null) {
            throw new IllegalArgumentException("Layer cannot be null");
        }
        if (layer instanceof DenseLayer) {
            DenseLayer denseLayer = (DenseLayer) layer;

            double[][] weightData = denseLayer.getWeights().getData();
            double[][] weightGradData = denseLayer.getWeightGradients().getData();
            for (int i = 0; i < weightData.length; i++) {
                for (int j = 0; j < weightData[i].length; j++) {
                    weightData[i][j] -= learningRate * weightGradData[i][j];
                }
            }

            double[][] biasData = denseLayer.getBiases().getData();
            double[][] biasGradData = denseLayer.getBiasGradients().getData();
            for (int i = 0; i < biasData.length; i++) {
                for (int j = 0; j < biasData[i].length; j++) {
                    biasData[i][j] -= learningRate * biasGradData[i][j];
                }
            }

            // Update the weights and biases in the layer
            denseLayer.setWeights(new Tensor(weightData));
            denseLayer.setBiases(new Tensor(biasData));
        }
    }

    public double getLearningRate() {
        return learningRate;
    }

    // EFFECTS: Sets learning rate;
    // throws IllegalArgumentException if learningRate is negative
    public void setLearningRate(double learningRate) {
        if (learningRate <= 0) {
            throw new IllegalArgumentException("Learning rate must be positive");
        }
        this.learningRate = learningRate;
    }
}
