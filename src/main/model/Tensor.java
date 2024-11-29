package model;

import org.json.JSONArray;
import org.json.JSONObject;

import persistence.Writable;

// Represents a multi-dimensional array of numbers used in neural networks.
public class Tensor implements Writable {

    private double[][] data;

    // EFFECTS: initializes this tensor with the given data;
    // throws IllegalArgumentException if data is null
    public Tensor(double[][] data) {
        if (data == null) {
            throw new IllegalArgumentException("Data cannot be null");
        }
        int rows = data.length;
        int cols = data[0].length;
        this.data = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            if (data[i].length != cols) {
                throw new IllegalArgumentException("All rows must have the same number of columns");
            }
            System.arraycopy(data[i], 0, this.data[i], 0, cols); // Faster copying
        }
    }

    // MODIFIES: this
    // EFFECTS: adds the elements of other to the elements of this tensor;
    // throws IllegalArgumentException if other is null or dimensions do not match
    public void add(Tensor other) {
        if (other == null) {
            EventLog.getInstance().logEvent(new Event("Attempted to add a null tensor."));
            throw new IllegalArgumentException("Other tensor cannot be null");
        }
        double[][] otherData = other.getData();
        if (data.length != otherData.length || data[0].length != otherData[0].length) {
            EventLog.getInstance().logEvent(new Event("Tensor dimension mismatch in add operation: dimensions "
                    + data.length + "x" + data[0].length + " and " + otherData.length + "x" + otherData[0].length));
            throw new IllegalArgumentException("Tensor dimensions must match");
        }
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[i].length; j++) {
                data[i][j] += otherData[i][j];
            }
        }
    }

    // MODIFIES: this
    // EFFECTS: multiplies the elements of this tensor with the elements of the other tensor;
    // throws IllegalArgumentException if other is null or dimensions do not match
    public void multiply(Tensor other) {
        if (other == null) {
            EventLog.getInstance().logEvent(new Event("Attempted to multiply by a null tensor."));
            throw new IllegalArgumentException("Other tensor cannot be null");
        }
        double[][] otherData = other.getData();
        if (data.length != otherData.length || data[0].length != otherData[0].length) {
            EventLog.getInstance().logEvent(new Event("Tensor dimension mismatch in multiply operation: dimensions "
                    + data.length + "x" + data[0].length + " and " + otherData.length + "x" + otherData[0].length));
            throw new IllegalArgumentException("Tensor dimensions must match");
        }
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[i].length; j++) {
                data[i][j] *= otherData[i][j];
            }
        }
    }

    // EFFECTS: returns the data of this tensor
    public double[][] getData() {
        int rows = data.length;
        int cols = data[0].length;
        double[][] copyData = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(data[i], 0, copyData[i], 0, cols);
        }
        return copyData;
    }

    @Override
    public JSONObject toJson() {
        JSONObject json = new JSONObject();
        JSONArray dataArray = new JSONArray();
        for (double[] row : data) {
            JSONArray rowArray = new JSONArray();
            for (double val : row) {
                rowArray.put(val);
            }
            dataArray.put(rowArray);
        }
        json.put("data", dataArray);
        EventLog.getInstance().logEvent(new Event("Serialized Tensor to JSON with dimensions "
                + data.length + "x" + data[0].length));
        return json;
    }

    // EFFECTS: Construct a Tensor from a JSONObject
    public static Tensor fromJson(JSONObject json) {
        JSONArray dataArray = json.getJSONArray("data");
        int rows = dataArray.length();
        int cols = dataArray.getJSONArray(0).length();
        double[][] data = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            JSONArray rowArray = dataArray.getJSONArray(i);
            for (int j = 0; j < cols; j++) {
                data[i][j] = rowArray.getDouble(j);
            }
        }
        EventLog.getInstance().logEvent(new Event("Deserialized Tensor from JSON with dimensions "
                + data.length + "x" + data[0].length));
        return new Tensor(data);
    }
}
