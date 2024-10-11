package model;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

public class TestSGDOptimizer {
    private SGDOptimizer optimizer;
    private DenseLayer layer;

    @BeforeEach
    void runBefore() {
        optimizer = new SGDOptimizer(0.01);
        layer = new DenseLayer(2, 3);
    }

    @Test
    void testConstructor() {
        assertEquals(0.01, optimizer.getLearningRate());
    }

    @Test
    void testConstructorWithInvalidLearningRate() {
        assertThrows(IllegalArgumentException.class, () -> {
            new SGDOptimizer(0);
        });
        assertThrows(IllegalArgumentException.class, () -> {
            new SGDOptimizer(-0.01);
        });
    }

    @Test
    void testUpdateParameters() {
        assertDoesNotThrow(() -> {
            optimizer.updateParameters(layer);
        });
    }

    @Test
    void testUpdateParametersWithNullLayer() {
        assertThrows(IllegalArgumentException.class, () -> {
            optimizer.updateParameters(null);
        });
    }
}
