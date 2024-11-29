package ui.gui;

import javax.swing.*;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import model.Event;
import model.EventLog;

/**
 * Main GUI class for the Neural Network application.
 */
public class NeuralNetworkGUI extends JFrame {
    private ApplicationController controller;

    // EFFECTS: Constructs the GUI for the application.
    public NeuralNetworkGUI() {
        controller = new ApplicationController();
        setTitle("JTorch Neural Network Application");
        setDefaultCloseOperation(JFrame.DO_NOTHING_ON_CLOSE);
        setSize(1000, 700);
        setLocationRelativeTo(null);

        addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent e) {
                printEventLog();
                dispose();
                System.exit(0);
            }
        });

        initializeComponents();
    }

    // MODIFIES: this
    // EFFECTS: Initializes the components of the GUI.
    private void initializeComponents() {
        JTabbedPane tabbedPane = new JTabbedPane();

        GraphPanel graphPanel = new GraphPanel();

        tabbedPane.addTab("Add Tensors", new AddTensorPanel(controller));
        tabbedPane.addTab("Add Layers", new AddLayerPanel(controller));
        tabbedPane.addTab("Train Network", new TrainPanel(controller, graphPanel));
        tabbedPane.addTab("Loss Graph", graphPanel);
        tabbedPane.addTab("Architecture", new ArchitecturePanel(controller));

        add(tabbedPane);
    }

    // EFFECTS: Prints all events from the EventLog to the console
    private void printEventLog() {
        EventLog eventLog = EventLog.getInstance();
        System.out.println("Event Log:");
        for (Event event : eventLog) {
            System.out.println(event.toString());
        }
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            NeuralNetworkGUI gui = new NeuralNetworkGUI();
            gui.setVisible(true);
        });
    }
}
