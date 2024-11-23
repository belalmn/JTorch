package ui.gui;

import model.Tensor;

import javax.swing.*;
import javax.swing.table.AbstractTableModel;
import java.awt.*;
import java.io.File;

/*
 * Panel for adding and managing tensors in the application.
 */
public class AddTensorPanel extends JPanel {
    private ApplicationController controller;
    private JTextField tensorNameField;
    private JTextField rowsField;
    private JTextField colsField;
    private JButton createButton;
    private JTable tensorTable;
    private JScrollPane scrollPane;
    private TensorTableModel tensorTableModel;
    private JPanel inputPanel;
    private JButton saveTensorButton;
    private JButton loadTensorButton;

    // MODIFIES: this
    // EFFECTS: Constructs a new AddTensorPanel with the given controller.
    public AddTensorPanel(ApplicationController controller) {
        this.controller = controller;
        setLayout(new BorderLayout());

        initializeComponents();
    }

    // MODIFIES: this
    // EFFECTS: Initializes the components of the panel.
    @SuppressWarnings("methodlength")
    private void initializeComponents() {
        inputPanel = new JPanel(new GridBagLayout());
        GridBagConstraints gbc = new GridBagConstraints();

        // Constraints for all components
        gbc.anchor = GridBagConstraints.WEST;
        gbc.insets = new Insets(5, 5, 5, 5);

        // Row 0: Tensor Name
        gbc.gridx = 0;
        gbc.gridy = 0;
        inputPanel.add(new JLabel("Tensor Name:"), gbc);

        gbc.gridx = 1;
        tensorNameField = new JTextField(15);
        inputPanel.add(tensorNameField, gbc);

        // Row 1: Rows
        gbc.gridx = 0;
        gbc.gridy = 1;
        inputPanel.add(new JLabel("Rows:"), gbc);

        gbc.gridx = 1;
        rowsField = new JTextField(5);
        inputPanel.add(rowsField, gbc);

        // Row 2: Columns
        gbc.gridx = 0;
        gbc.gridy = 2;
        inputPanel.add(new JLabel("Columns:"), gbc);

        gbc.gridx = 1;
        colsField = new JTextField(5);
        inputPanel.add(colsField, gbc);

        // Row 3: Buttons (Create, Save, Load)
        gbc.gridx = 0;
        gbc.gridy = 3;
        gbc.gridwidth = 2;
        gbc.anchor = GridBagConstraints.CENTER;

        JPanel buttonPanel = new JPanel(new FlowLayout(FlowLayout.CENTER, 10, 0));
        createButton = new JButton("Create Tensor");
        createButton.addActionListener(e -> createTensor());
        buttonPanel.add(createButton);

        saveTensorButton = new JButton("Save Tensor");
        saveTensorButton.addActionListener(e -> saveTensor());
        buttonPanel.add(saveTensorButton);

        loadTensorButton = new JButton("Load Tensor");
        loadTensorButton.addActionListener(e -> loadTensor());
        buttonPanel.add(loadTensorButton);

        inputPanel.add(buttonPanel, gbc);

        add(inputPanel, BorderLayout.NORTH);

        // Initialize tensor table
        tensorTableModel = new TensorTableModel();
        tensorTable = new JTable(tensorTableModel);
        tensorTable.setRowHeight(35);
        tensorTable.getColumn("Actions").setCellRenderer(new ButtonRenderer());
        tensorTable.getColumn("Actions").setCellEditor(new ButtonEditor(new JCheckBox()));

        scrollPane = new JScrollPane(tensorTable);
        add(scrollPane, BorderLayout.CENTER);
    }

    // MODIFIES: this
    // EFFECTS: Creates a new tensor with the given name, rows, and columns.
    @SuppressWarnings("methodlength")
    private void createTensor() {
        String tensorName = tensorNameField.getText().trim();
        if (tensorName.isEmpty()) {
            JOptionPane.showMessageDialog(this, "Please enter a tensor name.");
            return;
        }

        if (controller.getTensors().containsKey(tensorName)) {
            JOptionPane.showMessageDialog(
                    this, 
                    "A tensor with this name already exists. Please choose a different name."
            );
            return;
        }

        int rows;
        int cols;
        try {
            rows = Integer.parseInt(rowsField.getText().trim());
            cols = Integer.parseInt(colsField.getText().trim());
        } catch (NumberFormatException ex) {
            JOptionPane.showMessageDialog(this, "Please enter valid integers for rows and columns.");
            return;
        }

        // Open a dialog to input tensor data
        TensorDataDialog dataDialog = new TensorDataDialog(tensorName, rows, cols);
        dataDialog.setVisible(true);

        if (dataDialog.isConfirmed()) {
            double[][] data = dataDialog.getTensorData();
            Tensor tensor = new Tensor(data);
            controller.addTensor(tensorName, tensor);
            tensorTableModel.addTensor(tensorName);
            JOptionPane.showMessageDialog(this, "Tensor '" + tensorName + "' created successfully.");
            resetInputFields();
        }
    }

    // MODIFIES: this
    // EFFECTS: Resets the input fields to their default values.
    private void resetInputFields() {
        tensorNameField.setText("");
        rowsField.setText("");
        colsField.setText("");
    }

    // MODIFIES: this
    // EFFECTS: Saves the selected tensor to a file.
    private void saveTensor() {
        int selectedRow = tensorTable.getSelectedRow();
        if (selectedRow == -1) {
            JOptionPane.showMessageDialog(this, "Please select a tensor to save.");
            return;
        }
        String tensorName = (String) tensorTableModel.getValueAt(selectedRow, 0);
        Tensor tensor = controller.getTensors().get(tensorName);

        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setDialogTitle("Save Tensor");
        int userSelection = fileChooser.showSaveDialog(this);

        if (userSelection == JFileChooser.APPROVE_OPTION) {
            File fileToSave = fileChooser.getSelectedFile();
            try {
                controller.saveTensorToFile(tensor, fileToSave);
                JOptionPane.showMessageDialog(this, "Tensor saved successfully.");
            } catch (Exception e) {
                JOptionPane.showMessageDialog(this, "Error saving tensor: " + e.getMessage());
            }
        }
    }

    // MODIFIES: this
    // EFFECTS: Loads a tensor from a file.
    private void loadTensor() {
        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setDialogTitle("Load Tensor");
        int userSelection = fileChooser.showOpenDialog(this);

        if (userSelection == JFileChooser.APPROVE_OPTION) {
            File fileToOpen = fileChooser.getSelectedFile();
            String tensorName = JOptionPane.showInputDialog(this, "Enter a name for the loaded tensor:");
            if (tensorName == null || tensorName.trim().isEmpty()) {
                JOptionPane.showMessageDialog(this, "Tensor name cannot be empty.");
                return;
            }
            if (controller.getTensors().containsKey(tensorName)) {
                JOptionPane.showMessageDialog(this, "A tensor with this name already exists.");
                return;
            }
            try {
                Tensor tensor = controller.loadTensorFromFile(fileToOpen);
                controller.addTensor(tensorName, tensor);
                tensorTableModel.addTensor(tensorName);
                JOptionPane.showMessageDialog(this, "Tensor loaded successfully.");
            } catch (Exception e) {
                JOptionPane.showMessageDialog(this, "Error loading tensor: " + e.getMessage());
            }
        }
    }

    /*
     * Table model for the tensor table.
     */
    private class TensorTableModel extends AbstractTableModel {
        private final String[] columnNames = {"Tensor Name", "Actions"};
        private java.util.List<String> tensorNames;

        // MODIFIES: this
        // EFFECTS: Constructs a new TensorTableModel.
        public TensorTableModel() {
            tensorNames = new java.util.ArrayList<>(controller.getTensors().keySet());
        }

        // MODIFIES: this
        // EFFECTS: Adds a tensor to the table.
        public void addTensor(String tensorName) {
            tensorNames.add(tensorName);
            fireTableDataChanged();
        }

        // MODIFIES: this
        // EFFECTS: Removes a tensor from the table.
        public void removeTensor(String tensorName) {
            tensorNames.remove(tensorName);
            fireTableDataChanged();
        }

        @Override
        public int getRowCount() {
            return tensorNames.size();
        }

        @Override
        public int getColumnCount() {
            return columnNames.length;
        }

        @Override
        public String getColumnName(int columnIndex) {
            return columnNames[columnIndex];
        }

        @Override
        public Object getValueAt(int rowIndex, int columnIndex) {
            String tensorName = tensorNames.get(rowIndex);
            if (columnIndex == 0) {
                return tensorName;
            } else {
                return "View/Edit/Delete";
            }
        }

        @Override
        public boolean isCellEditable(int rowIndex, int columnIndex) {
            return columnIndex == 1; // Only the "Actions" column is editable (for buttons)
        }
    }

    /*
     * Renderer and editor for the "Actions" column in the tensor table.
     */
    private class ButtonRenderer extends JPanel implements javax.swing.table.TableCellRenderer {
        private JButton viewButton = new JButton("View");
        private JButton editButton = new JButton("Edit");
        private JButton deleteButton = new JButton("Delete");

        // MODIFIES: this
        // EFFECTS: Constructs a new ButtonRenderer.
        public ButtonRenderer() {
            setLayout(new FlowLayout(FlowLayout.CENTER));
            add(viewButton);
            add(editButton);
            add(deleteButton);
        }

        @Override
        public Component getTableCellRendererComponent(JTable table, Object value,
                                                      boolean isSelected, boolean hasFocus,
                                                      int row, int column) {
            return this;
        }
    }

    /*
     * Editor for the "Actions" column in the tensor table.
     */
    private class ButtonEditor extends AbstractCellEditor implements javax.swing.table.TableCellEditor {
        private JPanel panel;
        private JButton viewButton = new JButton("View");
        private JButton editButton = new JButton("Edit");
        private JButton deleteButton = new JButton("Delete");

        // MODIFIES: this
        // EFFECTS: Constructs a new ButtonEditor with the given check box.
        public ButtonEditor(JCheckBox checkBox) {
            panel = new JPanel(new FlowLayout(FlowLayout.CENTER));
            panel.add(viewButton);
            panel.add(editButton);
            panel.add(deleteButton);

            // Action listeners
            viewButton.addActionListener(e -> viewTensor());
            editButton.addActionListener(e -> editTensor());
            deleteButton.addActionListener(e -> deleteTensor());
        }

        // MODIFIES: this
        // EFFECTS: Opens a dialog to view the selected tensor.
        private void viewTensor() {
            int row = tensorTable.getSelectedRow();
            String tensorName = (String) tensorTable.getValueAt(row, 0);
            Tensor tensor = controller.getTensors().get(tensorName);

            // Show tensor data in a dialog
            TensorDataDialog viewDialog = new TensorDataDialog(tensorName, tensor.getData());
            viewDialog.setEditable(false);
            viewDialog.setVisible(true);
        }

        // MODIFIES: this
        // EFFECTS: Opens a dialog to edit the selected tensor.
        private void editTensor() {
            int row = tensorTable.getSelectedRow();
            String tensorName = (String) tensorTable.getValueAt(row, 0);
            Tensor tensor = controller.getTensors().get(tensorName);

            // Open tensor data in an editable dialog
            TensorDataDialog editDialog = new TensorDataDialog(tensorName, tensor.getData());
            editDialog.setEditable(true);
            editDialog.setVisible(true);

            if (editDialog.isConfirmed()) {
                double[][] newData = editDialog.getTensorData();
                Tensor newTensor = new Tensor(newData);
                controller.getTensors().put(tensorName, newTensor);
                JOptionPane.showMessageDialog(null, "Tensor '" + tensorName + "' updated successfully.");
            }
        }

        // MODIFIES: this
        // EFFECTS: Deletes the selected tensor.
        private void deleteTensor() {
            int row = tensorTable.getSelectedRow();
            String tensorName = (String) tensorTable.getValueAt(row, 0);

            int confirm = JOptionPane.showConfirmDialog(null,
                    "Are you sure you want to delete tensor '" + tensorName + "'?",
                    "Confirm Delete", JOptionPane.YES_NO_OPTION);

            if (confirm == JOptionPane.YES_OPTION) {
                controller.removeTensor(tensorName);
                tensorTableModel.removeTensor(tensorName);
                JOptionPane.showMessageDialog(null, "Tensor '" + tensorName + "' deleted successfully.");
            }
        }

        @Override
        public Object getCellEditorValue() {
            return "";
        }

        @Override
        public Component getTableCellEditorComponent(JTable table, Object value,
                                                     boolean isSelected, int row, int column) {
            return panel;
        }
    }
}
