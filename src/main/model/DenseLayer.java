package model;

import java.util.Random;

import org.json.JSONObject;

// Represents a fully connected (dense) layer in a neural network.
public class DenseLayer extends Layer {

    private Tensor weights;
    private Tensor biases;
    private Tensor inputCache; // Stores input for use in backward pass
    private Tensor weightGradients;
    private Tensor biasGradients;

    // EFFECTS: initializes weights and biases randomly;
    // throws IllegalArgumentException if inputSize <= 0 or outputSize <= 0
    public DenseLayer(int inputSize, int outputSize) {
        if (inputSize <= 0 || outputSize <= 0) {
            throw new IllegalArgumentException("Input and output sizes must be positive");
        }
        this.weights = initializeRandomTensor(inputSize, outputSize);
        this.biases = initializeRandomTensor(1, outputSize);
        EventLog.getInstance().logEvent(new Event("Dense layer initialized with input size " + inputSize
                + " and output size " + outputSize));
    }

    // EFFECTS: initializes weights and biases to given Tensor values;
    // throws IllegalArgumentException if weights and biases Tensors are null.
    public DenseLayer(Tensor weights, Tensor biases) {
        if (weights == null || biases == null) {
            throw new IllegalArgumentException("Weights and biases cannot be null");
        }
        this.weights = weights;
        this.biases = biases;
        EventLog.getInstance().logEvent(new Event("Dense layer initialized with preloaded weights and biases. "
                + getDescription()));
    }

    // MODIFIES: this
    // EFFECTS: computes output = input * weights + biases;
    // throws IllegalArgumentException if input is null or dimensions are invalid
    public Tensor forward(Tensor input) {
        if (input == null) {
            EventLog.getInstance().logEvent(new Event("Attempted forward pass with null input tensor in DenseLayer."));
            throw new IllegalArgumentException("Input cannot be null");
        }
        double[][] inputData = input.getData();
        double[][] weightData = weights.getData();
        double[][] biasData = biases.getData();

        if (inputData[0].length != weightData.length) {
            EventLog.getInstance()
                    .logEvent(new Event("Input dimensions do not match weights in DenseLayer forward pass: "
                            + "Input dimensions: " + inputData[0].length + ", Weight dimensions: "
                            + weightData.length));
            throw new IllegalArgumentException("Input dimensions do not match weights");
        }

        this.inputCache = input; // Store input for backpropagation

        int batchSize = inputData.length;
        int outputSize = weightData[0].length;
        double[][] outputData = new double[batchSize][outputSize];

        matMul(inputData, weightData, biasData, batchSize, outputSize, outputData);
        return new Tensor(outputData);
    }

    // Helper method to compute matrix multiplication of input, weights, and biases
    private void matMul(double[][] inputData, double[][] weightData, double[][] biasData, int batchSize,
            int outputSize, double[][] outputData) {
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                double sum = biasData[0][j];
                for (int k = 0; k < inputData[i].length; k++) {
                    sum += inputData[i][k] * weightData[k][j];
                }
                outputData[i][j] = sum;
            }
        }
    }

    // MODIFIES: this
    // EFFECTS: computes gradients with respect to weights, biases, and input;
    // throws IllegalArgumentException if gradient is null
    public Tensor backward(Tensor gradient) {
        if (gradient == null) {
            EventLog.getInstance()
                    .logEvent(new Event("Attempted backward pass with null gradient tensor in DenseLayer."));
            throw new IllegalArgumentException("Gradient cannot be null");
        }
        double[][] gradData = gradient.getData();
        double[][] inputData = inputCache.getData();
        double[][] weightData = weights.getData();

        int batchSize = inputData.length;
        int inputSize = inputData[0].length;
        int outputSize = gradData[0].length;

        // Compute gradients for weights and biases
        computeWeightAndBiasGradients(gradData, inputData, batchSize, inputSize, outputSize);

        // Compute gradient to pass to previous layer
        double[][] prevGradData = computePrevGradient(gradData, weightData, batchSize, inputSize, outputSize);

        return new Tensor(prevGradData);
    }

    // Helper method to compute weight and bias gradients
    private void computeWeightAndBiasGradients(double[][] gradData, double[][] inputData,
            int batchSize, int inputSize, int outputSize) {
        double[][] weightGradData = new double[inputSize][outputSize];
        double[][] biasGradData = new double[1][outputSize];

        for (int i = 0; i < batchSize; i++) {
            for (int k = 0; k < outputSize; k++) {
                biasGradData[0][k] += gradData[i][k];
                for (int j = 0; j < inputSize; j++) {
                    weightGradData[j][k] += inputData[i][j] * gradData[i][k];
                }
            }
        }
        this.weightGradients = new Tensor(weightGradData);
        this.biasGradients = new Tensor(biasGradData);
    }

    // Helper method to compute the gradient to pass to the previous layer
    private double[][] computePrevGradient(double[][] gradData, double[][] weightData,
            int batchSize, int inputSize, int outputSize) {
        double[][] prevGradData = new double[batchSize][inputSize];

        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                double sum = 0.0;
                for (int k = 0; k < outputSize; k++) {
                    sum += gradData[i][k] * weightData[j][k];
                }
                prevGradData[i][j] = sum;
            }
        }
        return prevGradData;
    }

    // MODIFIES: this
    // EFFECTS: updates weights and biases using computed gradients;
    // throws IllegalArgumentException if optimizer is null
    public void updateParameters(Optimizer optimizer) {
        if (optimizer == null) {
            EventLog.getInstance()
                    .logEvent(new Event("Attempted to update parameters with null optimizer in DenseLayer."));
            throw new IllegalArgumentException("Optimizer cannot be null");
        }
        optimizer.updateParameters(this);
    }

    // EFFECTS: returns a string representation of the layer
    public String getDescription() {
        return "Dense Layer (" + getInputSize() + " -> " + getOutputSize() + ")";
    }

    // Helper method to initialize tensors with random values
    private Tensor initializeRandomTensor(int rows, int cols) {
        double[][] data = new double[rows][cols];
        Random rand = new Random();
        double stdDev = 1.0 / Math.sqrt(rows);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = rand.nextGaussian() * stdDev;
            }
        }
        return new Tensor(data);
    }

    // Getters for weights, biases, and gradients
    public Tensor getWeights() {
        return weights;
    }

    public Tensor getBiases() {
        return biases;
    }

    public Tensor getWeightGradients() {
        return weightGradients;
    }

    public Tensor getBiasGradients() {
        return biasGradients;
    }

    public int getInputSize() {
        return weights.getData().length;
    }

    public int getOutputSize() {
        return weights.getData()[0].length;
    }

    // Setters for weights and biases
    public void setWeights(Tensor weights) {
        this.weights = weights;
    }

    public void setBiases(Tensor biases) {
        this.biases = biases;
    }

    @Override
    public JSONObject toJson() {
        JSONObject json = new JSONObject();
        json.put("type", "DenseLayer");
        json.put("weights", weights.toJson());
        json.put("biases", biases.toJson());
        EventLog.getInstance().logEvent(new Event("Serialized DenseLayer to JSON with input size "
                + weights.getData().length + " and output size " + weights.getData()[0].length));
        return json;
    }

    // EFFECTS: Construct DenseLayer from a JSONObject
    public static DenseLayer fromJson(JSONObject json) {
        Tensor weights = Tensor.fromJson(json.getJSONObject("weights"));
        Tensor biases = Tensor.fromJson(json.getJSONObject("biases"));
        EventLog.getInstance().logEvent(new Event("Deserialized DenseLayer from JSON with input size "
                + weights.getData().length + " and output size " + weights.getData()[0].length));
        return new DenseLayer(weights, biases);
    }
}
