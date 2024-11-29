package model;

import org.json.JSONObject;

// Implements the stochastic gradient descent (SGD) optimization algorithm.
public class SgdOptimizer extends Optimizer {

    private double learningRate;

    // EFFECTS: initializes the optimizer with the given learning rate;
    // throws IllegalArgumentException if learningRate <= 0
    public SgdOptimizer(double learningRate) {
        if (learningRate <= 0) {
            EventLog.getInstance().logEvent(
                    new Event("Attempted to initialize SgdOptimizer with invalid learning rate: " + learningRate));
            throw new IllegalArgumentException("Learning rate must be positive");
        }
        this.learningRate = learningRate;
        EventLog.getInstance().logEvent(new Event("Initialized SgdOptimizer with learning rate: " + learningRate));
    }

    // MODIFIES: layer
    // EFFECTS: updates the layer's parameters using SGD update rule;
    // throws IllegalArgumentException if layer is null
    @Override
    public void updateParameters(Layer layer) {
        if (layer == null) {
            EventLog.getInstance()
                    .logEvent(new Event("Attempted to update parameters with a null layer in SgdOptimizer."));
            throw new IllegalArgumentException("Layer cannot be null");
        }
        if (layer instanceof DenseLayer) {
            DenseLayer denseLayer = (DenseLayer) layer;

            double[][] weightData = denseLayer.getWeights().getData();
            double[][] weightGradData = denseLayer.getWeightGradients().getData();
            applyGradients(weightData, weightGradData);

            double[][] biasData = denseLayer.getBiases().getData();
            double[][] biasGradData = denseLayer.getBiasGradients().getData();
            applyGradients(biasData, biasGradData);

            // Update the weights and biases in the layer
            denseLayer.setWeights(new Tensor(weightData));
            denseLayer.setBiases(new Tensor(biasData));
        }
    }

    // Helper method to apply gradients to the data
    // MODIFIES: data
    // EFFECTS: applies the gradients to the data using the SGD update rule
    private void applyGradients(double[][] data, double[][] gradData) {
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[i].length; j++) {
                data[i][j] -= learningRate * gradData[i][j];
            }
        }
    }

    // EFFECTS: Sets learning rate;
    // throws IllegalArgumentException if learningRate is negative
    public void setLearningRate(double learningRate) {
        if (learningRate <= 0) {
            EventLog.getInstance()
                    .logEvent(new Event("Attempted to set invalid learning rate in SgdOptimizer: " + learningRate));
            throw new IllegalArgumentException("Learning rate must be positive");
        }
        this.learningRate = learningRate;
        EventLog.getInstance().logEvent(new Event("SgdOptimizer learning rate set to: " + learningRate));
    }

    public double getLearningRate() {
        return learningRate;
    }

    @Override
    public JSONObject toJson() {
        JSONObject json = new JSONObject();
        json.put("type", "SgdOptimizer");
        json.put("learningRate", learningRate);
        return json;
    }

    // EFFECTS: Construct an SgdOptimizer from a JSONObject
    public static SgdOptimizer fromJson(JSONObject json) {
        double learningRate = json.getDouble("learningRate");
        EventLog.getInstance()
                .logEvent(new Event("Deserialized SgdOptimizer from JSON with learning rate: " + learningRate));
        return new SgdOptimizer(learningRate);
    }
}
