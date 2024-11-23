package ui.gui;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.*;

import javax.swing.*;
import java.awt.*;

/**
 * Panel for displaying the training loss graph.
 */
public class GraphPanel extends JPanel {
    private XYSeries series;
    private XYSeriesCollection dataset;
    private JFreeChart chart;

    // MODIFIES: this
    // EFFECTS: Constructs a new GraphPanel.
    public GraphPanel() {
        series = new XYSeries("Loss");
        dataset = new XYSeriesCollection(series);
        chart = ChartFactory.createXYLineChart(
                "Training Loss",
                "Epoch",
                "Loss",
                dataset,
                PlotOrientation.VERTICAL,
                true,
                true,
                false
        );

        ChartPanel chartPanel = new ChartPanel(chart);
        setLayout(new BorderLayout());
        add(chartPanel, BorderLayout.CENTER);
    }

    // MODIFIES: this
    // EFFECTS: Updates the graph with the given epoch and loss.
    public void updateGraph(int epoch, double loss) {
        series.add(epoch, loss);
    }

    // MODIFIES: this
    // EFFECTS: Clears the data from the graph.
    public void clearData() {
        series.clear();
    }
}
