package model;

// Represents metrics for evaluating the performance of the neural network.
public class Metric {

    // EFFECTS: computes and returns the loss value (e.g., Mean Squared Error);
    // throws IllegalArgumentException if output or target is null or dimensions do not match
    public double calculateLoss(Tensor output, Tensor target) {
        if (output == null || target == null) {
            throw new IllegalArgumentException("Output and target cannot be null");
        }
        double[][] outputData = output.getData();
        double[][] targetData = target.getData();

        if (outputData.length != targetData.length || outputData[0].length != targetData[0].length) {
            throw new IllegalArgumentException("Output and target must have the same dimensions");
        }

        double loss = 0.0;
        int count = 0;
        for (int i = 0; i < outputData.length; i++) {
            for (int j = 0; j < outputData[i].length; j++) {
                double diff = outputData[i][j] - targetData[i][j];
                loss += diff * diff;
                count++;
            }
        }
        return loss / count;
    }

    // EFFECTS: computes and returns the accuracy;
    // throws IllegalArgumentException if output or target is null or dimensions do not match
    public double calculateAccuracy(Tensor output, Tensor target) {
        if (output == null || target == null) {
            throw new IllegalArgumentException("Output and target cannot be null");
        }
        double[][] outputData = output.getData();
        double[][] targetData = target.getData();

        if (outputData.length != targetData.length || outputData[0].length != targetData[0].length) {
            throw new IllegalArgumentException("Output and target must have the same dimensions");
        }

        int correct = 0;
        int total = 0;
        for (int i = 0; i < outputData.length; i++) {
            for (int j = 0; j < outputData[i].length; j++) {
                double outputValue = outputData[i][j] >= 0.5 ? 1.0 : 0.0;
                double targetValue = targetData[i][j];
                if (outputValue == targetValue) {
                    correct++;
                }
                total++;
            }
        }
        return (double) correct / total;
    }

    // EFFECTS: computes the gradient of the loss function with respect to the output;
    // throws IllegalArgumentException if output or target is null or dimensions do not match
    public Tensor lossGradient(Tensor output, Tensor target) {
        if (output == null || target == null) {
            throw new IllegalArgumentException("Output and target cannot be null");
        }
        double[][] outputData = output.getData();
        double[][] targetData = target.getData();

        if (outputData.length != targetData.length || outputData[0].length != targetData[0].length) {
            throw new IllegalArgumentException("Output and target must have the same dimensions");
        }

        double[][] gradData = new double[outputData.length][outputData[0].length];
        int count = outputData.length * outputData[0].length;

        for (int i = 0; i < outputData.length; i++) {
            for (int j = 0; j < outputData[i].length; j++) {
                gradData[i][j] = 2 * (outputData[i][j] - targetData[i][j]) / count;
            }
        }
        return new Tensor(gradData);
    }
}
