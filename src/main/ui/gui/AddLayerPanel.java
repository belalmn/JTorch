package ui.gui;

import model.Layer;

import javax.swing.*;
import javax.swing.table.AbstractTableModel;
import java.awt.*;
import java.io.File;

/**
 * Panel for adding and managing layers in the application.
 */
public class AddLayerPanel extends JPanel implements LayerChangeListener {
    private ApplicationController controller;
    private JButton addLayerButton;
    private JButton saveNetworkButton;
    private JButton loadNetworkButton;
    private JTable layerTable;
    private LayerTableModel layerTableModel;
    private JScrollPane scrollPane;

    // MODIFIES: this
    // EFFECTS: Constructs a new AddLayerPanel with the given controller.
    public AddLayerPanel(ApplicationController controller) {
        this.controller = controller;
        controller.addLayerChangeListener(this);
        setLayout(new BorderLayout());

        initializeComponents();
    }

    // MODIFIES: this
    // EFFECTS: Initializes the components of the panel.
    private void initializeComponents() {
        JPanel topPanel = new JPanel(new FlowLayout(FlowLayout.LEFT));

        // Button to add layer
        addLayerButton = new JButton("Add Layer");
        addLayerButton.addActionListener(e -> addLayer());
        topPanel.add(addLayerButton);

        // Save and Load buttons
        saveNetworkButton = new JButton("Save Network");
        saveNetworkButton.addActionListener(e -> saveNetwork());
        topPanel.add(saveNetworkButton);

        loadNetworkButton = new JButton("Load Network");
        loadNetworkButton.addActionListener(e -> loadNetwork());
        topPanel.add(loadNetworkButton);

        add(topPanel, BorderLayout.NORTH);

        // Initialize layer table
        layerTableModel = new LayerTableModel();
        layerTable = new JTable(layerTableModel);
        layerTable.setRowHeight(35);
        layerTable.getColumn("Actions").setCellRenderer(new LayerButtonRenderer());
        layerTable.getColumn("Actions").setCellEditor(new LayerButtonEditor(new JCheckBox()));

        scrollPane = new JScrollPane(layerTable);
        add(scrollPane, BorderLayout.CENTER);
    }

    // MODIFIES: this
    // EFFECTS: Adds a new layer to the neural network.
    private void addLayer() {
        // Open the LayerDataDialog for adding a new layer
        LayerDataDialog layerDialog = new LayerDataDialog(null);
        layerDialog.setVisible(true);

        if (layerDialog.isConfirmed()) {
            Layer layer = layerDialog.createLayer();
            if (layer != null) {
                controller.getNeuralNetwork().addLayer(layer);
                layerTableModel.updateLayers();
                JOptionPane.showMessageDialog(this, "Layer added successfully.");
                // Notify the architecture panel to update
                controller.notifyLayerListeners();
            }
        }
    }

    // MODIFIES: this
    // EFFECTS: Saves the neural network to a file.
    private void saveNetwork() {
        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setDialogTitle("Save Neural Network");
        int userSelection = fileChooser.showSaveDialog(this);

        if (userSelection == JFileChooser.APPROVE_OPTION) {
            File fileToSave = fileChooser.getSelectedFile();
            try {
                controller.saveNetworkToFile(fileToSave);
                JOptionPane.showMessageDialog(this, "Neural network saved successfully.");
            } catch (Exception e) {
                JOptionPane.showMessageDialog(this, "Error saving neural network: " + e.getMessage());
            }
        }
    }

    // MODIFIES: this
    // EFFECTS: Loads a neural network from a file.
    private void loadNetwork() {
        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setDialogTitle("Load Neural Network");
        int userSelection = fileChooser.showOpenDialog(this);

        if (userSelection == JFileChooser.APPROVE_OPTION) {
            File fileToOpen = fileChooser.getSelectedFile();
            try {
                controller.loadNetworkFromFile(fileToOpen);
                layerTableModel.updateLayers();
                JOptionPane.showMessageDialog(this, "Neural network loaded successfully.");
                controller.notifyLayerListeners();
            } catch (Exception e) {
                JOptionPane.showMessageDialog(this, "Error loading neural network: " + e.getMessage());
            }
        }
    }

    /**
     * Table model for displaying layers in a JTable.
     */
    private class LayerTableModel extends AbstractTableModel {
        private final String[] columnNames = {"Layer Index", "Layer Type", "Description", "Actions"};
        private java.util.List<Layer> layers;

        // MODIFIES: this
        // EFFECTS: Constructs a new LayerTableModel.
        public LayerTableModel() {
            layers = new java.util.ArrayList<>(controller.getNeuralNetwork().getLayers());
        }

        // MODIFIES: this
        // EFFECTS: Updates the layers in the table model.
        public void updateLayers() {
            layers = new java.util.ArrayList<>(controller.getNeuralNetwork().getLayers());
            fireTableDataChanged();
        }

        // EFFECTS: Returns the number of rows in the table.
        @Override
        public int getRowCount() {
            return layers.size();
        }

        // EFFECTS: Returns the number of columns in the table.
        @Override
        public int getColumnCount() {
            return columnNames.length;
        }

        // EFFECTS: Returns the name of the column at the given index.
        @Override
        public String getColumnName(int columnIndex) {
            return columnNames[columnIndex];
        }

        // EFFECTS: Returns the value at the given row and column index.
        @Override
        public Object getValueAt(int rowIndex, int columnIndex) {
            Layer layer = layers.get(rowIndex);
            switch (columnIndex) {
                case 0:
                    return rowIndex + 1;
                case 1:
                    return layer.getClass().getSimpleName();
                case 2:
                    return layer.getDescription();
                case 3:
                    return "View/Edit/Delete";
                default:
                    return "";
            }
        }

        // EFFECTS: Returns true if the cell at the given row and column index is editable.
        @Override
        public boolean isCellEditable(int rowIndex, int columnIndex) {
            return columnIndex == 3; // Only the Actions column is editable
        }
    }

    /**
     * Renderer for the "Actions" column in the layer table.
     */
    private class LayerButtonRenderer extends JPanel implements javax.swing.table.TableCellRenderer {
        private JButton viewButton = new JButton("View");
        private JButton editButton = new JButton("Edit");
        private JButton deleteButton = new JButton("Delete");

        // MODIFIES: this
        public LayerButtonRenderer() {
            setLayout(new FlowLayout(FlowLayout.CENTER));
            add(viewButton);
            add(editButton);
            add(deleteButton);
        }

        // EFFECTS: Returns the component for rendering the cell.
        @Override
        public Component getTableCellRendererComponent(JTable table, Object value,
                                                      boolean isSelected, boolean hasFocus,
                                                      int row, int column) {
            return this;
        }
    }

    /**
     * Editor for the "Actions" column in the layer table.
     */
    private class LayerButtonEditor extends AbstractCellEditor implements javax.swing.table.TableCellEditor {
        private JPanel panel;
        private JButton viewButton = new JButton("View");
        private JButton editButton = new JButton("Edit");
        private JButton deleteButton = new JButton("Delete");

        // MODIFIES: this
        // EFFECTS: Constructs a new LayerButtonEditor with the given check box.
        public LayerButtonEditor(JCheckBox checkBox) {
            panel = new JPanel(new FlowLayout(FlowLayout.CENTER));
            panel.add(viewButton);
            panel.add(editButton);
            panel.add(deleteButton);

            // Add action listeners
            viewButton.addActionListener(e -> viewLayer());
            editButton.addActionListener(e -> editLayer());
            deleteButton.addActionListener(e -> deleteLayer());
        }

        // MODIFIES: this
        // EFFECTS: Opens a dialog to view the layer details.
        private void viewLayer() {
            int row = layerTable.getSelectedRow();
            Layer layer = controller.getNeuralNetwork().getLayers().get(row);

            // Show layer details in a dialog
            LayerDataDialog viewDialog = new LayerDataDialog(layer);
            viewDialog.setEditable(false);
            viewDialog.setVisible(true);
        }

        // MODIFIES: this
        // EFFECTS: Opens a dialog to edit the layer details.
        private void editLayer() {
            int row = layerTable.getSelectedRow();
            Layer layer = controller.getNeuralNetwork().getLayers().get(row);

            // Open layer data in an editable dialog
            LayerDataDialog editDialog = new LayerDataDialog(layer);
            editDialog.setEditable(true);
            editDialog.setVisible(true);

            if (editDialog.isConfirmed()) {
                Layer newLayer = editDialog.createLayer();
                if (newLayer != null) {
                    // Replace the layer in the neural network
                    controller.getNeuralNetwork().getLayers().set(row, newLayer);
                    layerTableModel.updateLayers();
                    JOptionPane.showMessageDialog(null, "Layer updated successfully.");
                    controller.notifyLayerListeners();
                }
            }
        }

        // MODIFIES: this
        // EFFECTS: Deletes the selected layer.
        private void deleteLayer() {
            int row = layerTable.getSelectedRow();

            int confirm = JOptionPane.showConfirmDialog(null,
                    "Are you sure you want to delete this layer?",
                    "Confirm Delete", JOptionPane.YES_NO_OPTION);

            if (confirm == JOptionPane.YES_OPTION) {
                controller.getNeuralNetwork().getLayers().remove(row);
                layerTableModel.updateLayers();
                JOptionPane.showMessageDialog(null, "Layer deleted successfully.");
                controller.notifyLayerListeners();
            }
        }

        // EFFECTS: Returns the value of the cell editor.
        @Override
        public Object getCellEditorValue() {
            // No value to return since the editor is a button
            return "";
        }

        // EFFECTS: Returns the component for editing the cell.
        @Override
        public Component getTableCellEditorComponent(JTable table, Object value,
                                                     boolean isSelected, int row, int column) {
            return panel;
        }
    }

    // EFFECTS: Updates the architecture panel when the layer list changes.
    @Override
    public void onLayerListChanged() {
        layerTableModel.updateLayers();
    }
}
