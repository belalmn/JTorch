package model;

import java.util.List;

import org.json.JSONArray;
import org.json.JSONObject;

import persistence.Writable;

import java.util.ArrayList;

// Represents a neural network composed of multiple layers.
public class NeuralNetwork implements Writable {

    private List<Layer> layers;
    private TrainingListener trainingListener;

    // EFFECTS: initializes an empty list of layers
    public NeuralNetwork() {
        layers = new ArrayList<>();
    }

    // MODIFIES: this
    // EFFECTS: adds the layer to the network;
    // throws IllegalArgumentException if layer is null
    public void addLayer(Layer layer) {
        if (layer == null) {
            EventLog.getInstance().logEvent(new Event("Attempted to add a null layer to the network."));
            throw new IllegalArgumentException("Layer cannot be null");
        }
        layers.add(layer);
        EventLog.getInstance().logEvent(new Event("Added layer: " + layer.getDescription()
                + " to the network. Network now has " + layers.size() + " layers."));
    }

    // MODIFIES: this
    // EFFECTS: updates the layer at the specified index from the network with the new layer;
    // throws IllegalArgumentException if index is invalid or layer is null
    public void updateLayer(int index, Layer layer) {
        if (layer == null) {
            EventLog.getInstance().logEvent(new Event("Attempted to update layer with null layer."));
            throw new IllegalArgumentException("Layer cannot be null");
        }
        if (index < 0 || index >= layers.size()) {
            EventLog.getInstance().logEvent(new Event("Attempted to update layer at invalid index: " + index));
            throw new IllegalArgumentException("Invalid index");
        }
        layers.set(index, layer);
        EventLog.getInstance().logEvent(new Event("Updated layer at index " + index + " with new layer: "
                + layer.getDescription()));
    }

    // MODIFIES: this
    // EFFECTS: removes the layer at the specified index from the network;
    // throws IllegalArgumentException if index is invalid
    public void removeLayer(int index) {
        if (index < 0 || index >= layers.size()) {
            EventLog.getInstance().logEvent(new Event("Attempted to remove layer at invalid index: " + index));
            throw new IllegalArgumentException("Invalid index");
        }
        layers.remove(index);
        EventLog.getInstance().logEvent(new Event("Removed layer at index " + index + ". Network now has "
                + layers.size() + " layers."));
    }

    // MODIFIES: this
    // EFFECTS: sets the training listener
    public void setTrainingListener(TrainingListener listener) {
        this.trainingListener = listener;
    }

    // MODIFIES: this
    // EFFECTS: trains the network on the data for the specified number of epochs;
    // throws IllegalArgumentException if inputs or targets are null,
    // sizes do not match, epochs <= 0, or optimizer is null;
    // notifies UI of loss for each epoch throughout training
    public void train(List<Tensor> inputs, List<Tensor> targets, int epochs, Optimizer optimizer) {
        if (inputs == null || targets == null || optimizer == null || epochs <= 0 || inputs.size() != targets.size()) {
            EventLog.getInstance().logEvent(new Event("Invalid training parameters provided."));
            throw new IllegalArgumentException("Invalid training parameters");
        }
        EventLog.getInstance().logEvent(new Event("Training started for " + epochs + " epochs with optimizer: "
                + optimizer.getClass().getSimpleName()));
        Metric metric = new Metric();
        for (int epoch = 0; epoch < epochs; epoch++) {
            trainEpoch(inputs, targets, optimizer, metric);
            double totalLoss = trainEpoch(inputs, targets, optimizer, metric);
            double averageLoss = totalLoss / inputs.size();

            EventLog.getInstance().logEvent(
                    new Event("Epoch " + (epoch + 1) + "/" + epochs + " completed. Average Loss: " + averageLoss));

            // Notifies listener of new epoch and loss
            if (trainingListener != null) {
                trainingListener.onEpochEnd(epoch + 1, epochs, averageLoss);
            }
        }
        EventLog.getInstance().logEvent(new Event("Training completed after " + epochs + " epochs."));
    }

    // Helper method to train for one epoch
    private double trainEpoch(List<Tensor> inputs, List<Tensor> targets, Optimizer optimizer, Metric metric) {
        double totalLoss = 0;
        for (int i = 0; i < inputs.size(); i++) {
            Tensor output = forwardPass(inputs.get(i));
            double loss = metric.calculateLoss(output, targets.get(i));
            totalLoss += loss;
            Tensor lossGradient = metric.lossGradient(output, targets.get(i));
            backwardPass(lossGradient);
            updateParameters(optimizer);
        }
        return totalLoss;
    }

    // Helper method for the forward pass
    private Tensor forwardPass(Tensor input) {
        Tensor output = input;
        for (Layer layer : layers) {
            output = layer.forward(output);
        }
        return output;
    }

    // Helper method for the backward pass
    private void backwardPass(Tensor lossGradient) {
        Tensor grad = lossGradient;
        for (int j = layers.size() - 1; j >= 0; j--) {
            grad = layers.get(j).backward(grad);
        }
    }

    // Helper method to update parameters of all layers
    private void updateParameters(Optimizer optimizer) {
        for (Layer layer : layers) {
            layer.updateParameters(optimizer);
        }
    }

    // EFFECTS: computes the output of the network for the given input;
    // throws IllegalArgumentException if input is null
    public Tensor predict(Tensor input) {
        if (input == null) {
            EventLog.getInstance().logEvent(new Event("Attempted to predict with null input tensor."));
            throw new IllegalArgumentException("Input cannot be null");
        }
        return forwardPass(input);
    }

    // EFFECTS: returns a string listing the layers and their configurations
    public String getArchitecture() {
        StringBuilder sb = new StringBuilder();
        for (Layer layer : layers) {
            sb.append(layer.getClass().getSimpleName());
            if (layer instanceof DenseLayer) {
                DenseLayer denseLayer = (DenseLayer) layer;
                sb.append(" (input size: ").append(denseLayer.getWeights().getData().length);
                sb.append(", output size: ").append(denseLayer.getWeights().getData()[0].length).append(")");
            } else if (layer instanceof ActivationLayer) {
                ActivationLayer activationLayer = (ActivationLayer) layer;
                sb.append(" (activation: ").append(activationLayer.getActivationFunction()).append(")");
            }
            sb.append("\n");
        }
        return sb.toString();
    }

    // Getter for the list of layers (added for testing purposes)
    public List<Layer> getLayers() {
        return layers;
    }

    @Override
    public JSONObject toJson() {
        JSONObject json = new JSONObject();
        JSONArray layersArray = new JSONArray();
        for (Layer layer : layers) {
            layersArray.put(layer.toJson());
        }
        json.put("layers", layersArray);
        return json;
    }

    // EFFECTS: Construct NeuralNetwork from a JSONObject
    public static NeuralNetwork fromJson(JSONObject json) {
        NeuralNetwork nn = new NeuralNetwork();
        JSONArray layersArray = json.getJSONArray("layers");
        for (int i = 0; i < layersArray.length(); i++) {
            JSONObject layerJson = layersArray.getJSONObject(i);
            String type = layerJson.getString("type");
            Layer layer = null;
            if (type.equals("DenseLayer")) {
                layer = DenseLayer.fromJson(layerJson);
            } else if (type.equals("ActivationLayer")) {
                layer = ActivationLayer.fromJson(layerJson);
            }
            if (layer != null) {
                nn.addLayer(layer);
            }
        }
        EventLog.getInstance()
                .logEvent(new Event("Deserialized NeuralNetwork from JSON with " + nn.layers.size() + " layers."));
        return nn;
    }
}
