package ui.gui;

import javax.swing.*;
import java.awt.*;

/**
 * Dialog for displaying and editing tensor data.
 */
public class TensorDataDialog extends JDialog {
    private String tensorName;
    private int rows;
    private int cols;
    private double[][] tensorData;
    private JTable dataTable;
    private boolean confirmed = false;
    private boolean editable = true;

    // MODIFIES: this
    // EFFECTS: Constructs a new TensorDataDialog with the given tensor name and dimensions.
    public TensorDataDialog(String tensorName, int rows, int cols) {
        this.tensorName = tensorName;
        this.rows = rows;
        this.cols = cols;
        this.tensorData = new double[rows][cols];
        initializeComponents();
    }

    // MODIFIES: this
    // EFFECTS: Constructs a new TensorDataDialog with the given tensor name and data.
    public TensorDataDialog(String tensorName, double[][] data) {
        this.tensorName = tensorName;
        this.rows = data.length;
        this.cols = data[0].length;
        this.tensorData = data;
        initializeComponents();
    }

    // MODIFIES: this
    // EFFECTS: Initializes the components of the dialog.
    @SuppressWarnings("methodlength")
    private void initializeComponents() {
        setTitle("Tensor Data - " + tensorName);
        setModal(true);
        setSize(500, 400);
        setLocationRelativeTo(null);

        String[] columnNames = new String[cols];
        for (int i = 0; i < cols; i++) {
            columnNames[i] = "Col " + (i + 1);
        }

        Object[][] tableData = new Object[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                tableData[i][j] = tensorData[i][j];
            }
        }

        dataTable = new JTable(tableData, columnNames);
        dataTable.setEnabled(editable);

        JScrollPane scrollPane = new JScrollPane(dataTable);
        add(scrollPane, BorderLayout.CENTER);

        if (editable) {
            JPanel buttonPanel = new JPanel();
            JButton okButton = new JButton("OK");
            JButton cancelButton = new JButton("Cancel");
            buttonPanel.add(okButton);
            buttonPanel.add(cancelButton);

            okButton.addActionListener(e -> onOk());
            cancelButton.addActionListener(e -> onCancel());

            add(buttonPanel, BorderLayout.SOUTH);
        } else {
            JButton closeButton = new JButton("Close");
            closeButton.addActionListener(e -> dispose());
            add(closeButton, BorderLayout.SOUTH);
        }
    }

    // MODIFIES: this
    // EFFECTS: Sets whether the tensor data is editable.
    public void setEditable(boolean editable) {
        this.editable = editable;
        dataTable.setEnabled(editable);
    }

    // MODIFIES: this
    // EFFECTS: Handles the OK button click event.
    private void onOk() {
        try {
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    Object value = dataTable.getValueAt(i, j);
                    if (value == null) {
                        throw new NumberFormatException("Cell at (" + (i + 1) + ", " + (j + 1) + ") is empty.");
                    }
                    tensorData[i][j] = Double.parseDouble(value.toString());
                }
            }
            confirmed = true;
            dispose();
        } catch (NumberFormatException ex) {
            JOptionPane.showMessageDialog(this, "Invalid data in tensor table: " + ex.getMessage());
        }
    }

    // MODIFIES: this
    // EFFECTS: Handles the Cancel button click event.
    private void onCancel() {
        confirmed = false;
        dispose();
    }

    public boolean isConfirmed() {
        return confirmed;
    }

    public double[][] getTensorData() {
        return tensorData;
    }
}
