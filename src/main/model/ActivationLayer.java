package model;

import org.json.JSONObject;

// Represents an activation layer in a neural network.
public class ActivationLayer extends Layer {

    private String activationFunction;
    private Tensor inputCache; // Stores input for use in backward pass

    // EFFECTS: initializes the activation layer with the specified function;
    // throws IllegalArgumentException if activationFunction is null or unsupported
    public ActivationLayer(String activationFunction) {
        if (activationFunction == null || (
                !activationFunction.equalsIgnoreCase("relu") 
                && !activationFunction.equalsIgnoreCase("sigmoid"))
            ) {
            throw new IllegalArgumentException("Unsupported activation function");
        }
        this.activationFunction = activationFunction.toLowerCase();
    }

    // MODIFIES: this
    // EFFECTS: applies activation function to input tensor;
    // throws IllegalArgumentException if input is null
    public Tensor forward(Tensor input) {
        if (input == null) {
            throw new IllegalArgumentException("Input cannot be null");
        }
        this.inputCache = input; // Store input for backpropagation

        double[][] inputData = input.getData();
        double[][] outputData = new double[inputData.length][inputData[0].length];

        for (int i = 0; i < inputData.length; i++) {
            for (int j = 0; j < inputData[i].length; j++) {
                double x = inputData[i][j];
                if (activationFunction.equals("relu")) {
                    outputData[i][j] = Math.max(0, x);
                } else if (activationFunction.equals("sigmoid")) {
                    outputData[i][j] = 1 / (1 + Math.exp(-x));
                }
            }
        }
        return new Tensor(outputData);
    }

    // EFFECTS: computes gradient of activation function and multiplies element-wise;
    // throws IllegalArgumentException if gradient is null
    public Tensor backward(Tensor gradient) {
        if (gradient == null) {
            throw new IllegalArgumentException("Gradient cannot be null");
        }

        double[][] gradData = gradient.getData();
        double[][] inputData = inputCache.getData();
        double[][] outputGradData = new double[gradData.length][gradData[0].length];

        for (int i = 0; i < gradData.length; i++) {
            for (int j = 0; j < gradData[i].length; j++) {
                double x = inputData[i][j];
                double derivative = 0.0;
                if (activationFunction.equals("relu")) {
                    derivative = x > 0 ? 1 : 0;
                } else if (activationFunction.equals("sigmoid")) {
                    double sigmoid = 1 / (1 + Math.exp(-x));
                    derivative = sigmoid * (1 - sigmoid);
                }
                outputGradData[i][j] = gradData[i][j] * derivative;
            }
        }
        return new Tensor(outputGradData);
    }

    // EFFECTS: does nothing as activation layers typically have no parameters;
    // throws IllegalArgumentException if optimizer is null
    public void updateParameters(Optimizer optimizer) {
        if (optimizer == null) {
            throw new IllegalArgumentException("Optimizer cannot be null");
        }
    }

    public String getActivationFunction() {
        return activationFunction;
    }

    @Override
    public JSONObject toJson() {
        JSONObject json = new JSONObject();
        json.put("type", "ActivationLayer");
        json.put("activationFunction", activationFunction);
        return json;
    }

    // EFFECTS: Construct ActivationLayer from a JSONObject
    public static ActivationLayer fromJson(JSONObject json) {
        String activationFunction = json.getString("activationFunction");
        return new ActivationLayer(activationFunction);
    }
}
