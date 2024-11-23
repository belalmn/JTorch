package ui.gui;

import model.Layer;
import model.NeuralNetwork;

import javax.swing.*;
import java.awt.*;

/**
 * Panel for displaying the architecture of the neural network.
 */
public class ArchitecturePanel extends JPanel implements LayerChangeListener {
    private ApplicationController controller;
    private DefaultListModel<String> architectureListModel;
    private JList<String> architectureList;

    // MODIFIES: this
    // EFFECTS: Constructs a new ArchitecturePanel with the given controller.
    public ArchitecturePanel(ApplicationController controller) {
        this.controller = controller;
        controller.addLayerChangeListener(this);
        setLayout(new BorderLayout());

        initializeComponents();
        updateArchitecture();
    }

    // MODIFIES: this
    // EFFECTS: Initializes the components of the panel.
    private void initializeComponents() {
        architectureListModel = new DefaultListModel<>();
        architectureList = new JList<>(architectureListModel);
        add(new JScrollPane(architectureList), BorderLayout.CENTER);
    }

    // MODIFIES: this
    // EFFECTS: Updates the architecture list with the current layers in the neural network.
    public void updateArchitecture() {
        architectureListModel.clear();
        NeuralNetwork neuralNetwork = controller.getNeuralNetwork();
        int layerIndex = 1;
        for (Layer layer : neuralNetwork.getLayers()) {
            architectureListModel.addElement("Layer " + layerIndex + ": " + layer.getDescription());
            layerIndex++;
        }
    }

    // MODIFIES: this
    // EFFECTS: Called when the layer list has changed.
    @Override
    public void onLayerListChanged() {
        updateArchitecture();
    }
}