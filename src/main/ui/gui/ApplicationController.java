package ui.gui;

import model.NeuralNetwork;
import model.Optimizer;
import model.SgdOptimizer;
import model.Tensor;
import org.json.JSONObject;
import persistence.JsonReader;
import persistence.JsonWriter;

import java.io.File;
import java.io.IOException;
import java.util.*;

/**
 * Controller class for the application.
 */
public class ApplicationController {
    private NeuralNetwork neuralNetwork;
    private Map<String, Tensor> tensors;
    private List<TensorChangeListener> tensorListeners;
    private List<LayerChangeListener> layerListeners;

    // MODIFIES: this
    // EFFECTS: Constructs the application controller.
    public ApplicationController() {
        neuralNetwork = new NeuralNetwork();
        tensors = new HashMap<>();
        tensorListeners = new ArrayList<>();
        layerListeners = new ArrayList<>();
    }

    // MODIFIES: this
    // EFFECTS: Adds a tensor change listener to the list of listeners.
    public void addTensorChangeListener(TensorChangeListener listener) {
        tensorListeners.add(listener);
    }

    // MODIFIES: this
    // EFFECTS: Removes a tensor change listener from the list of listeners.
    public void removeTensorChangeListener(TensorChangeListener listener) {
        tensorListeners.remove(listener);
    }

    // MODIFIES: this
    // EFFECTS: Notifies all tensor change listeners that the tensor list has changed.
    private void notifyTensorListeners() {
        for (TensorChangeListener listener : tensorListeners) {
            listener.onTensorListChanged();
        }
    }

    // MODIFIES: this
    // EFFECTS: Adds a layer change listener to the list of listeners.
    public void addLayerChangeListener(LayerChangeListener listener) {
        layerListeners.add(listener);
    }

    // MODIFIES: this
    // EFFECTS: Removes a layer change listener from the list of listeners.
    public void removeLayerChangeListener(LayerChangeListener listener) {
        layerListeners.remove(listener);
    }

    // MODIFIES: this
    // EFFECTS: Notifies all layer change listeners that the layer list has changed.
    public void notifyLayerListeners() {
        for (LayerChangeListener listener : layerListeners) {
            listener.onLayerListChanged();
        }
    }

    // EFFECTS: Sets the neural network to the given neural network.
    public void saveTensorToFile(Tensor tensor, File file) throws IOException {
        JsonWriter jsonWriter = new JsonWriter(file.getAbsolutePath());
        jsonWriter.open();
        jsonWriter.write(tensor);
        jsonWriter.close();
    }

    // EFFECTS: Loads a tensor from the given file.
    public Tensor loadTensorFromFile(File file) throws IOException {
        JsonReader jsonReader = new JsonReader(file.getAbsolutePath());
        JSONObject json = jsonReader.readJson();
        return Tensor.fromJson(json);
    }

    // EFFECTS: Saves the neural network to the given file.
    public void saveNetworkToFile(File file) throws IOException {
        JsonWriter jsonWriter = new JsonWriter(file.getAbsolutePath());
        jsonWriter.open();
        jsonWriter.write(neuralNetwork);
        jsonWriter.close();
    }
    
    // EFFECTS: Loads a neural network from the given file.
    public void loadNetworkFromFile(File file) throws IOException {
        JsonReader jsonReader = new JsonReader(file.getAbsolutePath());
        JSONObject json = jsonReader.readJson();
        neuralNetwork = NeuralNetwork.fromJson(json);
        notifyLayerListeners();
    }

    // EFFECTS: Saves the optimizer to the given file.
    public void saveOptimizerToFile(Optimizer optimizer, File file) throws IOException {
        JsonWriter jsonWriter = new JsonWriter(file.getAbsolutePath());
        jsonWriter.open();
        jsonWriter.write(optimizer);
        jsonWriter.close();
    }

    // EFFECTS: Loads an optimizer from the given file.
    public Optimizer loadOptimizerFromFile(File file) throws IOException {
        JsonReader jsonReader = new JsonReader(file.getAbsolutePath());
        JSONObject json = jsonReader.readJson();
        String type = json.getString("type");
        if (type.equals("SgdOptimizer")) {
            return SgdOptimizer.fromJson(json);
        } else {
            throw new IllegalArgumentException("Unsupported optimizer type: " + type);
        }
    }

    // MODIFIES: this
    // EFFECTS: Adds a layer to the neural network.
    public void addTensor(String name, Tensor tensor) {
        tensors.put(name, tensor);
        notifyTensorListeners();
    }

    // MODIFIES: this
    // EFFECTS: Removes a tensor from the list of tensors.
    public void removeTensor(String name) {
        tensors.remove(name);
        notifyTensorListeners();
    }
    
    public NeuralNetwork getNeuralNetwork() {
        return neuralNetwork;
    }

    public Map<String, Tensor> getTensors() {
        return tensors;
    }
}
