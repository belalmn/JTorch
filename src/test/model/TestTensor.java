package model;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

public class TestTensor {
    private Tensor tensor1;
    private Tensor tensor2;
    private Tensor tensor3;
    private double[][] data1;
    private double[][] data2;
    private double[][] data3;
    
    @BeforeEach
    void runBefore() {
        data1 = new double[][]{{1.0, 2.0}, {3.0, 4.0}};
        data2 = new double[][]{{5.0, 6.0}, {7.0, 8.0}};
        data3 = new double[][]{{1.0, 2.0, 3.0}};
        tensor1 = new Tensor(data1);
        tensor2 = new Tensor(data2);
    }

    @Test
    void testConstructor() {
        assertArrayEquals(data1, tensor1.getData());
    }

    @Test
    void testConstructorWithNullData() {
        assertThrows(IllegalArgumentException.class, () -> {
            new Tensor(null);
        });
    }

    @Test
    void testAdd() {
        tensor1.add(tensor2);
        double[][] expectedData = {{6.0, 8.0}, {10.0, 12.0}};
        assertArrayEquals(expectedData, tensor1.getData());
    }

    @Test
    void testAddWithNullOther() {
        assertThrows(IllegalArgumentException.class, () -> {
            tensor1.add(null);
        });
    }

    @Test
    void testAddWithMismatchedDimensions() {
        tensor3 = new Tensor(data3);
        assertThrows(IllegalArgumentException.class, () -> {
            tensor1.add(tensor3);
        });
    }

    @Test
    void testMultiply() {
        tensor1.multiply(tensor2);
        double[][] expectedData = {{5.0, 12.0}, {21.0, 32.0}};
        assertArrayEquals(expectedData, tensor1.getData());
    }

    @Test
    void testMultiplyWithNullOther() {
        assertThrows(IllegalArgumentException.class, () -> {
            tensor1.multiply(null);
        });
    }

    @Test
    void testMultiplyWithMismatchedDimensions() {
        tensor3 = new Tensor(data3);
        assertThrows(IllegalArgumentException.class, () -> {
            tensor1.multiply(tensor3);
        });
    }
}
