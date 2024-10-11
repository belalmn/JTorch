package model;

// Represents a multi-dimensional array of numbers used in neural networks.
public class Tensor {

    private double[][] data;

    // EFFECTS: initializes this tensor with the given data;
    // throws IllegalArgumentException if data is null
    public Tensor(double[][] data) {
        // TODO: implement constructor
    }

    // MODIFIES: this
    // EFFECTS: adds the elements of other to the elements of this tensor;
    // throws IllegalArgumentException if other is null or dimensions do not match
    public void add(Tensor other) {
        // TODO: implement add
    }

    // MODIFIES: this
    // EFFECTS: multiplies the elements of this tensor with the elements of the other tensor;
    // throws IllegalArgumentException if other is null or dimensions do not match
    public void multiply(Tensor other) {
        // TODO: implement multiply
    }

    // EFFECTS: returns the data of this tensor
    public double[][] getData() {
        return null; // stub
    }
}
