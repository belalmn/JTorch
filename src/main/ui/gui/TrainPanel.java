package ui.gui;

import model.NeuralNetwork;
import model.Tensor;
import model.Optimizer;
import model.SgdOptimizer;
import model.TrainingListener;

import javax.swing.*;
import java.awt.*;
import java.io.File;
import java.util.Arrays;
import java.util.List;

/**
 * Panel for training the neural network.
 */
public class TrainPanel extends JPanel implements TrainingListener, TensorChangeListener {
    private ApplicationController controller;
    private GraphPanel graphPanel;
    private JTextField epochsField;
    private JButton trainButton;
    private JProgressBar progressBar;
    private JComboBox<String> inputTensorComboBox;
    private JComboBox<String> targetTensorComboBox;
    private JComboBox<String> optimizerComboBox;
    private JTextField learningRateField;
    private JButton saveOptimizerButton;
    private JButton loadOptimizerButton;

    // MODIFIES: this
    // EFFECTS: Initializes the TrainPanel with the given controller and graphPanel.
    public TrainPanel(ApplicationController controller, GraphPanel graphPanel) {
        this.controller = controller;
        this.graphPanel = graphPanel;
        controller.addTensorChangeListener(this);
        initializeComponents();
    }

    // MODIFIES: this
    // EFFECTS: Initializes the components of the panel.
    @SuppressWarnings("methodlength")
    private void initializeComponents() {
        setLayout(new BorderLayout());

        // Initialize input panel with GridBagLayout
        JPanel inputPanel = new JPanel(new GridBagLayout());
        GridBagConstraints gbc = new GridBagConstraints();

        // Constraints for all components
        gbc.anchor = GridBagConstraints.WEST;
        gbc.insets = new Insets(5, 5, 5, 5);

        // Row 0: Epochs
        gbc.gridx = 0;
        gbc.gridy = 0;
        inputPanel.add(new JLabel("Epochs:"), gbc);

        gbc.gridx = 1;
        epochsField = new JTextField(10);
        inputPanel.add(epochsField, gbc);

        // Row 1: Input Tensor
        gbc.gridx = 0;
        gbc.gridy = 1;
        inputPanel.add(new JLabel("Input Tensor:"), gbc);

        gbc.gridx = 1;
        inputTensorComboBox = new JComboBox<>();
        inputPanel.add(inputTensorComboBox, gbc);

        // Row 2: Target Tensor
        gbc.gridx = 0;
        gbc.gridy = 2;
        inputPanel.add(new JLabel("Target Tensor:"), gbc);

        gbc.gridx = 1;
        targetTensorComboBox = new JComboBox<>();
        inputPanel.add(targetTensorComboBox, gbc);

        // Row 3: Optimizer
        gbc.gridx = 0;
        gbc.gridy = 3;
        inputPanel.add(new JLabel("Optimizer:"), gbc);

        gbc.gridx = 1;
        optimizerComboBox = new JComboBox<>(new String[]{"SGD"});
        inputPanel.add(optimizerComboBox, gbc);

        // Row 4: Learning Rate
        gbc.gridx = 0;
        gbc.gridy = 4;
        inputPanel.add(new JLabel("Learning Rate:"), gbc);

        gbc.gridx = 1;
        learningRateField = new JTextField(10);
        inputPanel.add(learningRateField, gbc);

        // Row 5: Save and Load Optimizer Buttons
        gbc.gridx = 0;
        gbc.gridy = 5;
        gbc.gridwidth = 2;
        gbc.anchor = GridBagConstraints.CENTER;

        JPanel buttonPanel = new JPanel(new FlowLayout(FlowLayout.CENTER, 10, 0));
        saveOptimizerButton = new JButton("Save Optimizer");
        saveOptimizerButton.addActionListener(e -> saveOptimizer());
        buttonPanel.add(saveOptimizerButton);

        loadOptimizerButton = new JButton("Load Optimizer");
        loadOptimizerButton.addActionListener(e -> loadOptimizer());
        buttonPanel.add(loadOptimizerButton);

        inputPanel.add(buttonPanel, gbc);

        add(inputPanel, BorderLayout.NORTH);

        // Center Panel: Progress Bar
        progressBar = new JProgressBar(0, 100);
        progressBar.setStringPainted(true);
        JPanel centerPanel = new JPanel(new BorderLayout());
        centerPanel.add(progressBar, BorderLayout.CENTER);
        add(centerPanel, BorderLayout.CENTER);

        // South Panel: Train Button
        trainButton = new JButton("Train Network");
        trainButton.addActionListener(e -> startTraining());
        add(trainButton, BorderLayout.SOUTH);

        // Populate the combo boxes with tensor names
        updateTensorComboBoxes();
    }

    // MODIFIES: this
    // EFFECTS: Updates the input and target tensor combo boxes with the latest tensor names.
    private void updateTensorComboBoxes() {
        inputTensorComboBox.removeAllItems();
        targetTensorComboBox.removeAllItems();

        for (String tensorName : controller.getTensors().keySet()) {
            inputTensorComboBox.addItem(tensorName);
            targetTensorComboBox.addItem(tensorName);
        }
    }

    // MODIFIES: this
    // EFFECTS: Starts training the neural network with the given settings.
    @SuppressWarnings("methodlength")
    private void startTraining() {
        String epochsText = epochsField.getText().trim();
        if (epochsText.isEmpty()) {
            JOptionPane.showMessageDialog(this, "Please enter the number of epochs.");
            return;
        }

        int epochs;
        try {
            epochs = Integer.parseInt(epochsText);
        } catch (NumberFormatException ex) {
            JOptionPane.showMessageDialog(this, "Please enter a valid integer for epochs.");
            return;
        }

        String inputTensorName = (String) inputTensorComboBox.getSelectedItem();
        String targetTensorName = (String) targetTensorComboBox.getSelectedItem();

        if (inputTensorName == null || targetTensorName == null) {
            JOptionPane.showMessageDialog(this, "Please select both input and target tensors.");
            return;
        }

        Tensor inputTensor = controller.getTensors().get(inputTensorName);
        Tensor targetTensor = controller.getTensors().get(targetTensorName);

        // Get optimizer settings
        String optimizerType = (String) optimizerComboBox.getSelectedItem();
        Optimizer optimizer = null;
        if (optimizerType.equals("SGD")) {
            String learningRateText = learningRateField.getText().trim();
            if (learningRateText.isEmpty()) {
                JOptionPane.showMessageDialog(this, "Please enter a learning rate.");
                return;
            }
            double learningRate;
            try {
                learningRate = Double.parseDouble(learningRateText);
            } catch (NumberFormatException ex) {
                JOptionPane.showMessageDialog(this, "Please enter a valid number for learning rate.");
                return;
            }
            optimizer = new SgdOptimizer(learningRate);
        } else {
            JOptionPane.showMessageDialog(this, "Unsupported optimizer type.");
            return;
        }

        NeuralNetwork neuralNetwork = controller.getNeuralNetwork();
        neuralNetwork.setTrainingListener(this);

        trainButton.setEnabled(false);
        progressBar.setValue(0);
        graphPanel.clearData();

        final Optimizer optimizerFinal = optimizer;

        // Start training in a separate thread
        SwingWorker<Void, Void> worker = new SwingWorker<Void, Void>() {
            @Override
            protected Void doInBackground() throws Exception {
                List<Tensor> inputs = Arrays.asList(inputTensor);
                List<Tensor> targets = Arrays.asList(targetTensor);

                neuralNetwork.train(inputs, targets, epochs, optimizerFinal);
                return null;
            }

            @Override
            protected void done() {
                trainButton.setEnabled(true);
                JOptionPane.showMessageDialog(null, "Training completed!");
            }
        };

        worker.execute();
    }

    // MODIFIES: this
    // EFFECTS: Saves the configured optimizer to a file.
    private void saveOptimizer() {
        String optimizerType = (String) optimizerComboBox.getSelectedItem();
        if (optimizerType == null) {
            JOptionPane.showMessageDialog(this, "Please select an optimizer to save.");
            return;
        }

        Optimizer optimizer = getConfiguredOptimizer();
        if (optimizer == null) {
            return; // Error message already shown
        }

        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setDialogTitle("Save Optimizer");
        int userSelection = fileChooser.showSaveDialog(this);

        if (userSelection == JFileChooser.APPROVE_OPTION) {
            File fileToSave = fileChooser.getSelectedFile();
            try {
                controller.saveOptimizerToFile(optimizer, fileToSave);
                JOptionPane.showMessageDialog(this, "Optimizer saved successfully.");
            } catch (Exception e) {
                JOptionPane.showMessageDialog(this, "Error saving optimizer: " + e.getMessage());
            }
        }
    }

    // MODIFIES: this
    // EFFECTS: Loads an optimizer from a file and updates the optimizer settings.
    private void loadOptimizer() {
        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setDialogTitle("Load Optimizer");
        int userSelection = fileChooser.showOpenDialog(this);

        if (userSelection == JFileChooser.APPROVE_OPTION) {
            File fileToOpen = fileChooser.getSelectedFile();
            try {
                Optimizer optimizer = controller.loadOptimizerFromFile(fileToOpen);
                if (optimizer instanceof SgdOptimizer) {
                    optimizerComboBox.setSelectedItem("SGD");
                    learningRateField.setText(String.valueOf(((SgdOptimizer) optimizer).getLearningRate()));
                } else {
                    JOptionPane.showMessageDialog(this, "Unsupported optimizer type.");
                }
                JOptionPane.showMessageDialog(this, "Optimizer loaded successfully.");
            } catch (Exception e) {
                JOptionPane.showMessageDialog(this, "Error loading optimizer: " + e.getMessage());
            }
        }
    }

    // EFFECTS: Returns the optimizer configured based on the user input.
    private Optimizer getConfiguredOptimizer() {
        String optimizerType = (String) optimizerComboBox.getSelectedItem();
        if (optimizerType.equals("SGD")) {
            String learningRateText = learningRateField.getText().trim();
            if (learningRateText.isEmpty()) {
                JOptionPane.showMessageDialog(this, "Please enter a learning rate.");
                return null;
            }
            double learningRate;
            try {
                learningRate = Double.parseDouble(learningRateText);
            } catch (NumberFormatException ex) {
                JOptionPane.showMessageDialog(this, "Please enter a valid number for learning rate.");
                return null;
            }
            return new SgdOptimizer(learningRate);
        } else {
            JOptionPane.showMessageDialog(this, "Unsupported optimizer type.");
            return null;
        }
    }

    // EFFECTS: Updates the progress bar and graph with the latest training epoch and loss.
    @Override
    public void onEpochEnd(int epoch, int totalEpochs, double loss) {
        SwingUtilities.invokeLater(() -> {
            // Update progress bar
            int progress = (int) ((epoch / (double) totalEpochs) * 100);
            progressBar.setValue(progress);

            // Update graph
            graphPanel.updateGraph(epoch, loss);
        });
    }

    // EFFECTS: Updates the tensor combo boxes when a tensor is added or removed.
    @Override
    public void onTensorListChanged() {
        updateTensorComboBoxes();
    }
}
