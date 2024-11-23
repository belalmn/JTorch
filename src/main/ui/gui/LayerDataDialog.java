package ui.gui;

import model.ActivationLayer;
import model.DenseLayer;
import model.Layer;

import javax.swing.*;
import java.awt.*;

/**
 * Dialog for entering data for a new layer.
 */
public class LayerDataDialog extends JDialog {
    private JComboBox<String> layerTypeCombo;
    private JButton okButton;
    private JButton cancelButton;
    private boolean confirmed = false;
    private boolean editable = true;
    private Layer existingLayer;

    // Panels for different layer types
    private JPanel denseLayerPanel;
    private JPanel activationLayerPanel;
    private JPanel emptyPanel;

    // Fields for DenseLayer
    private JTextField inputSizeField;
    private JTextField outputSizeField;

    // Fields for ActivationLayer
    private JTextField activationFunctionField;

    private CardLayout cardLayout;
    private JPanel centerPanel;

    // MODIFIES: this
    // EFFECTS: Constructs a new LayerDataDialog with the given layer.
    public LayerDataDialog(Layer layer) {
        this.existingLayer = layer;
        this.editable = layer == null || editable;
        setTitle(layer == null ? "Add Layer" : (editable ? "Edit Layer" : "View Layer"));
        setModal(true);
        setSize(400, 250);
        setLocationRelativeTo(null);

        initializeComponents();
        if (layer != null) {
            populateFields(layer);
        } else {
            // Set default selection to placeholder if adding a new layer
            layerTypeCombo.setSelectedIndex(0);
            updateFields();
        }
    }

    // MODIFIES: this
    // EFFECTS: Initializes the components of the dialog.
    @SuppressWarnings("methodlength")
    private void initializeComponents() {
        setLayout(new BorderLayout());

        // Layer type selection
        JPanel topPanel = new JPanel(new GridLayout(1, 2));
        topPanel.add(new JLabel("Layer Type:"));
        layerTypeCombo = new JComboBox<>(new String[]{"Select Layer Type", "Dense Layer", "Activation Layer"});
        layerTypeCombo.addActionListener(e -> updateFields());
        topPanel.add(layerTypeCombo);
        layerTypeCombo.setEnabled(existingLayer == null);
        add(topPanel, BorderLayout.NORTH);

        // Center panel with CardLayout
        cardLayout = new CardLayout();
        centerPanel = new JPanel(cardLayout);

        // Empty panel
        emptyPanel = new JPanel();

        // DenseLayer panel
        denseLayerPanel = new JPanel(new GridLayout(2, 2));
        inputSizeField = new JTextField();
        outputSizeField = new JTextField();

        denseLayerPanel.add(new JLabel("Input Size:"));
        denseLayerPanel.add(inputSizeField);
        denseLayerPanel.add(new JLabel("Output Size:"));
        denseLayerPanel.add(outputSizeField);

        // ActivationLayer panel
        activationLayerPanel = new JPanel(new GridLayout(1, 2));
        activationFunctionField = new JTextField();

        activationLayerPanel.add(new JLabel("Activation Function:"));
        activationLayerPanel.add(activationFunctionField);

        // Add panels to centerPanel
        centerPanel.add(emptyPanel, "Empty");
        centerPanel.add(denseLayerPanel, "Dense Layer");
        centerPanel.add(activationLayerPanel, "Activation Layer");

        add(centerPanel, BorderLayout.CENTER);

        // Buttons
        JPanel buttonPanel = new JPanel();
        okButton = new JButton("OK");
        cancelButton = new JButton(editable ? "Cancel" : "Close");
        buttonPanel.add(okButton);
        buttonPanel.add(cancelButton);

        okButton.addActionListener(e -> onOk());
        cancelButton.addActionListener(e -> onCancel());

        if (!editable) {
            okButton.setVisible(false);
        }

        add(buttonPanel, BorderLayout.SOUTH);
    }

    // MODIFIES: this
    // EFFECTS: Populates the fields of the dialog with the given layer.
    private void populateFields(Layer layer) {
        if (layer instanceof DenseLayer) {
            layerTypeCombo.setSelectedItem("Dense Layer");
            DenseLayer denseLayer = (DenseLayer) layer;
            inputSizeField.setText(String.valueOf(denseLayer.getInputSize()));
            outputSizeField.setText(String.valueOf(denseLayer.getOutputSize()));
        } else if (layer instanceof ActivationLayer) {
            layerTypeCombo.setSelectedItem("Activation Layer");
            ActivationLayer activationLayer = (ActivationLayer) layer;
            activationFunctionField.setText(activationLayer.getActivationFunction());
        }
        updateFields();
    }

    // MODIFIES: this
    // EFFECTS: Updates the fields based on the selected layer type.
    private void updateFields() {
        String selectedLayerType = (String) layerTypeCombo.getSelectedItem();

        switch (selectedLayerType) {
            case "Dense Layer":
                cardLayout.show(centerPanel, "Dense Layer");
                break;
            case "Activation Layer":
                cardLayout.show(centerPanel, "Activation Layer");
                break;
            default:
                cardLayout.show(centerPanel, "Empty");
                break;
        }

        // Set fields editable or not
        boolean fieldsEditable = editable && !selectedLayerType.equals("Select Layer Type");
        inputSizeField.setEditable(fieldsEditable);
        outputSizeField.setEditable(fieldsEditable);
        activationFunctionField.setEditable(fieldsEditable);
    }

    // MODIFIES: this
    // EFFECTS: Sets whether the dialog is editable or not.
    public void setEditable(boolean editable) {
        this.editable = editable;
        layerTypeCombo.setEnabled(false); // Disable changing layer type when editing/viewing
        okButton.setVisible(editable);
        cancelButton.setText(editable ? "Cancel" : "Close");
        updateFields();
    }

    // MODIFIES: this
    // EFFECTS: Handles the OK button click event.
    private void onOk() {
        // Validate inputs and set confirmed to true
        if (validateInputs()) {
            confirmed = true;
            dispose();
        }
    }

    // MODIFIES: this
    // EFFECTS: Handles the Cancel button click event.
    private void onCancel() {
        confirmed = false;
        dispose();
    }

    // EFFECTS: Returns true if the dialog was confirmed, false otherwise.
    public boolean isConfirmed() {
        return confirmed;
    }

    // EFFECTS: Creates a new layer based on the entered data.
    public Layer createLayer() {
        String selectedLayerType = (String) layerTypeCombo.getSelectedItem();
        try {
            if (selectedLayerType.equals("Dense Layer")) {
                int inputSize = Integer.parseInt(inputSizeField.getText().trim());
                int outputSize = Integer.parseInt(outputSizeField.getText().trim());
                return new DenseLayer(inputSize, outputSize);
            } else if (selectedLayerType.equals("Activation Layer")) {
                String activationFunction = activationFunctionField.getText().trim();
                if (activationFunction.isEmpty()) {
                    throw new IllegalArgumentException("Activation function cannot be empty.");
                }
                return new ActivationLayer(activationFunction);
            }
        } catch (NumberFormatException ex) {
            JOptionPane.showMessageDialog(this, "Please enter valid numbers for input/output sizes.");
        } catch (IllegalArgumentException ex) {
            JOptionPane.showMessageDialog(this, ex.getMessage());
        }
        return null;
    }

    // MODIFIES: this
    // EFFECTS: Validates the inputs and shows an error message if invalid.
    private boolean validateInputs() {
        String selectedLayerType = (String) layerTypeCombo.getSelectedItem();

        if (selectedLayerType.equals("Select Layer Type")) {
            JOptionPane.showMessageDialog(this, "Please select a layer type.");
            return false;
        }

        if (selectedLayerType.equals("Dense Layer")) {
            if (inputSizeField.getText().trim().isEmpty() || outputSizeField.getText().trim().isEmpty()) {
                JOptionPane.showMessageDialog(this, "Please enter input and output sizes.");
                return false;
            }
        } else if (selectedLayerType.equals("Activation Layer")) {
            if (activationFunctionField.getText().trim().isEmpty()) {
                JOptionPane.showMessageDialog(this, "Please enter an activation function.");
                return false;
            }
        }
        return true;
    }
}
