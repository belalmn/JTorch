package model;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

public class TestMetric {
    private Metric metric;
    private Tensor outputTensor;
    private Tensor targetTensor;

    @BeforeEach
    void runBefore() {
        metric = new Metric();
        double[][] outputData = {{0.8}};
        double[][] targetData = {{1.0}};
        outputTensor = new Tensor(outputData);
        targetTensor = new Tensor(targetData);
    }

    @Test
    void testCalculateLoss() {
        double loss = metric.calculateLoss(outputTensor, targetTensor);
        assertTrue(loss >= 0);
    }

    @Test
    void testCalculateLossWithNullOutput() {
        assertThrows(IllegalArgumentException.class, () -> {
            metric.calculateLoss(null, targetTensor);
        });
    }

    @Test
    void testCalculateLossWithNullTarget() {
        assertThrows(IllegalArgumentException.class, () -> {
            metric.calculateLoss(outputTensor, null);
        });
    }

    @Test
    void testCalculateLossWithMismatchedDimensions() {
        Tensor mismatchedTensor = new Tensor(new double[][]{{1.0, 2.0}});
        assertThrows(IllegalArgumentException.class, () -> {
            metric.calculateLoss(outputTensor, mismatchedTensor);
        });
    }

    @Test
    void testCalculateAccuracy() {
        double accuracy = metric.calculateAccuracy(outputTensor, targetTensor);
        assertTrue(accuracy >= 0 && accuracy <= 1);
    }

    @Test
    void testCalculateAccuracyWithNullOutput() {
        assertThrows(IllegalArgumentException.class, () -> {
            metric.calculateAccuracy(null, targetTensor);
        });
    }

    @Test
    void testCalculateAccuracyWithNullTarget() {
        assertThrows(IllegalArgumentException.class, () -> {
            metric.calculateAccuracy(outputTensor, null);
        });
    }

    @Test
    void testCalculateAccuracyWithMismatchedDimensions() {
        Tensor mismatchedTensor = new Tensor(new double[][]{{1.0, 2.0}});
        assertThrows(IllegalArgumentException.class, () -> {
            metric.calculateAccuracy(outputTensor, mismatchedTensor);
        });
    }

    @Test
    void testLossGradient() {
        Tensor gradient = metric.lossGradient(outputTensor, targetTensor);
        assertNotNull(gradient);
    }

    @Test
    void testLossGradientWithNullOutput() {
        assertThrows(IllegalArgumentException.class, () -> {
            metric.lossGradient(null, targetTensor);
        });
    }

    @Test
    void testLossGradientWithNullTarget() {
        assertThrows(IllegalArgumentException.class, () -> {
            metric.lossGradient(outputTensor, null);
        });
    }

    @Test
    void testLossGradientWithMismatchedDimensions() {
        Tensor mismatchedTensor = new Tensor(new double[][]{{1.0, 2.0}});
        assertThrows(IllegalArgumentException.class, () -> {
            metric.lossGradient(outputTensor, mismatchedTensor);
        });
    }
}
