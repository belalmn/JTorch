package ui;

import model.*;
import java.util.*;

/**
 * Represents the console application for managing a neural network.
 */
public class NeuralNetworkApp {

    private Scanner scanner;
    private NeuralNetwork neuralNetwork;
    private Optimizer optimizer;
    private Map<String, Tensor> tensors;

    // initializes the application
    public NeuralNetworkApp() {
        scanner = new Scanner(System.in);
        neuralNetwork = new NeuralNetwork();
        optimizer = null;
        tensors = new HashMap<>();
    }

    // runs the application loop
    public void run() {
        boolean running = true;
        while (running) {
            displayMenu();
            int choice = getIntegerInput("Enter your choice: ");
            switch (choice) {
                case 1:
                    createTensor();
                    break;
                case 2:
                    performTensorOperations();
                    break;
                case 3:
                    addLayer();
                    break;
                case 4:
                    chooseOptimizer();
                    break;
                case 5:
                    trainNetwork();
                    break;
                case 6:
                    viewArchitecture();
                    break;
                case 7:
                    running = false;
                    exitApplication();
                    break;
                default:
                    System.out.println("Invalid choice. Please try again.");
            }
        }
    }

    // displays the main menu
    private void displayMenu() {
        System.out.println("\nNeural Network Application Menu:");
        System.out.println("1. Create a tensor");
        System.out.println("2. Perform operations on tensors");
        System.out.println("3. Add a layer to the neural network");
        System.out.println("4. Choose optimizer");
        System.out.println("5. Train the neural network");
        System.out.println("6. View neural network architecture");
        System.out.println("7. Exit");
    }

    // allows the user to create a tensor
    private void createTensor() {
        String tensorName = getStringInput("Enter a name for the tensor: ");
        System.out.println("Enter the number of rows: ");
        int rows = getIntegerInput("");
        System.out.println("Enter the number of columns: ");
        int cols = getIntegerInput("");

        double[][] data = new double[rows][cols];
        System.out.println("Enter the elements row by row (space-separated):");
        for (int i = 0; i < rows; i++) {
            System.out.print("Row " + (i + 1) + ": ");
            String[] rowElements = scanner.nextLine().trim().split("\\s+");
            if (rowElements.length != cols) {
                System.out.println("Incorrect number of elements. Please re-enter the row.");
                i--;
                continue;
            }
            for (int j = 0; j < cols; j++) {
                try {
                    data[i][j] = Double.parseDouble(rowElements[j]);
                } catch (NumberFormatException e) {
                    System.out.println("Invalid number format. Please re-enter the row.");
                    i--;
                    break;
                }
            }
        }
        try {
            Tensor tensor = new Tensor(data);
            tensors.put(tensorName, tensor);
            System.out.println("Tensor '" + tensorName + "' created successfully.");
        } catch (IllegalArgumentException e) {
            System.out.println("Failed to create tensor: " + e.getMessage());
        }
    }

    // allows the user to perform addition or multiplication on tensors
    private void performTensorOperations() {
        if (tensors.isEmpty()) {
            System.out.println("No tensors available. Please create tensors first.");
            return;
        }
        System.out.println("Available tensors:");
        for (String name : tensors.keySet()) {
            System.out.println("- " + name);
        }
        String tensorName1 = getStringInput("Enter the name of the first tensor: ");
        String tensorName2 = getStringInput("Enter the name of the second tensor: ");
        if (!tensors.containsKey(tensorName1) || !tensors.containsKey(tensorName2)) {
            System.out.println("One or both tensors not found.");
            return;
        }
        Tensor tensor1 = tensors.get(tensorName1);
        Tensor tensor2 = tensors.get(tensorName2);

        System.out.println("Choose operation:");
        System.out.println("1. Addition");
        System.out.println("2. Multiplication");
        int choice = getIntegerInput("Enter your choice: ");
        try {
            if (choice == 1) {
                tensor1.add(tensor2);
                System.out.println("Result after addition stored in tensor '" + tensorName1 + "'.");
            } else if (choice == 2) {
                tensor1.multiply(tensor2);
                System.out.println("Result after multiplication stored in tensor '" + tensorName1 + "'.");
            } else {
                System.out.println("Invalid operation choice.");
            }
        } catch (IllegalArgumentException e) {
            System.out.println("Operation failed: " + e.getMessage());
        }
    }

    // allows the user to add a layer to the neural network
    private void addLayer() {
        System.out.println("Choose layer type to add:");
        System.out.println("1. Dense Layer");
        System.out.println("2. Activation Layer");
        int choice = getIntegerInput("Enter your choice: ");
        if (choice == 1) {
            int inputSize = getIntegerInput("Enter input size: ");
            int outputSize = getIntegerInput("Enter output size: ");
            try {
                Layer denseLayer = new DenseLayer(inputSize, outputSize);
                neuralNetwork.addLayer(denseLayer);
                System.out.println("Dense layer added to the neural network.");
            } catch (IllegalArgumentException e) {
                System.out.println("Failed to add layer: " + e.getMessage());
            }
        } else if (choice == 2) {
            String activationFunction = getStringInput("Enter activation function (relu/sigmoid): ");
            try {
                Layer activationLayer = new ActivationLayer(activationFunction);
                neuralNetwork.addLayer(activationLayer);
                System.out.println("Activation layer added to the neural network.");
            } catch (IllegalArgumentException e) {
                System.out.println("Failed to add layer: " + e.getMessage());
            }
        } else {
            System.out.println("Invalid layer type choice.");
        }
    }

    // allows the user to choose and configure an optimizer
    private void chooseOptimizer() {
        System.out.println("Choose optimizer:");
        System.out.println("1. Stochastic Gradient Descent (SGD)");
        int choice = getIntegerInput("Enter your choice: ");
        if (choice == 1) {
            double learningRate = getDoubleInput("Enter learning rate: ");
            try {
                optimizer = new SGDOptimizer(learningRate);
                System.out.println("SGD optimizer selected.");
            } catch (IllegalArgumentException e) {
                System.out.println("Failed to create optimizer: " + e.getMessage());
            }
        } else {
            System.out.println("Invalid optimizer choice.");
        }
    }

    // allows the user to train the neural network
    private void trainNetwork() {
        if (optimizer == null) {
            System.out.println("Please choose an optimizer before training.");
            return;
        }
        if (tensors.isEmpty()) {
            System.out.println("No tensors available. Please create tensors first.");
            return;
        }
        int epochs = getIntegerInput("Enter number of epochs: ");

        // Assuming the user provides input and target tensors from the available tensors
        System.out.println("Available tensors:");
        for (String name : tensors.keySet()) {
            System.out.println("- " + name);
        }

        String inputTensorName = getStringInput("Enter the name of the input tensor: ");
        String targetTensorName = getStringInput("Enter the name of the target tensor: ");
        if (!tensors.containsKey(inputTensorName) || !tensors.containsKey(targetTensorName)) {
            System.out.println("One or both tensors not found.");
            return;
        }

        List<Tensor> inputs = new ArrayList<>();
        inputs.add(tensors.get(inputTensorName));
        List<Tensor> targets = new ArrayList<>();
        targets.add(tensors.get(targetTensorName));

        System.out.println("Training started...");
        try {
            neuralNetwork.train(inputs, targets, epochs, optimizer);
            System.out.println("Training completed.");
        } catch (IllegalArgumentException e) {
            System.out.println("Training failed: " + e.getMessage());
        }
    }

    // displays the neural network's architecture
    private void viewArchitecture() {
        String architecture = neuralNetwork.getArchitecture();
        System.out.println("Neural Network Architecture:");
        System.out.println(architecture);
    }

    // exits the application
    private void exitApplication() {
        System.out.println("Exiting the application. Goodbye!");
        scanner.close();
    }

    // prompts the user for an integer input and returns it
    private int getIntegerInput(String prompt) {
        while (true) {
            System.out.print(prompt);
            String input = scanner.nextLine();
            try {
                int value = Integer.parseInt(input.trim());
                return value;
            } catch (NumberFormatException e) {
                System.out.println("Invalid integer. Please try again.");
            }
        }
    }

    // prompts the user for a double input and returns it
    private double getDoubleInput(String prompt) {
        while (true) {
            System.out.print(prompt);
            String input = scanner.nextLine();
            try {
                double value = Double.parseDouble(input.trim());
                return value;
            } catch (NumberFormatException e) {
                System.out.println("Invalid number. Please try again.");
            }
        }
    }

    // prompts the user for a string input and returns it
    private String getStringInput(String prompt) {
        System.out.print(prompt);
        return scanner.nextLine().trim();
    }

    // Main method to run the application
    public static void main(String[] args) {
        NeuralNetworkApp app = new NeuralNetworkApp();
        app.run();
    }
}
