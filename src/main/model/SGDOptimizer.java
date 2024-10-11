package model;

// Implements the stochastic gradient descent (SGD) optimization algorithm.
public class SGDOptimizer implements Optimizer {

    private double learningRate;

    // EFFECTS: initializes the optimizer with the given learning rate;
    // throws IllegalArgumentException if learningRate <= 0
    public SGDOptimizer(double learningRate) {
        if (learningRate <= 0) {
            throw new IllegalArgumentException("Learning rate must be positive");
        }
        this.learningRate = learningRate;
    }

    // MODIFIES: layer
    // EFFECTS: updates the layer's parameters using SGD update rule;
    // throws IllegalArgumentException if layer is null
    public void updateParameters(Layer layer) {
        if (layer == null) {
            throw new IllegalArgumentException("Layer cannot be null");
        }
        if (layer instanceof DenseLayer) {
            DenseLayer denseLayer = (DenseLayer) layer;
            Tensor weights = denseLayer.getWeights();
            Tensor biases = denseLayer.getBiases();
            Tensor weightGradients = denseLayer.getWeightGradients();
            Tensor biasGradients = denseLayer.getBiasGradients();

            double[][] weightData = weights.getData();
            double[][] weightGradData = weightGradients.getData();
            for (int i = 0; i < weightData.length; i++) {
                for (int j = 0; j < weightData[i].length; j++) {
                    weightData[i][j] -= learningRate * weightGradData[i][j];
                }
            }

            double[][] biasData = biases.getData();
            double[][] biasGradData = biasGradients.getData();
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

    public void setLearningRate(double learningRate) {
        if (learningRate <= 0) {
            throw new IllegalArgumentException("Learning rate must be positive");
        }
        this.learningRate = learningRate;
    }
}
