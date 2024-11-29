# JTorch - CPSC 210 Personal Project
## Description
### What will the application do?

The intention for this project is to implement a small **Deep Learning** framework, and apply it towards a specific task, such as recognizing handwritten digits and matching them to the right value. This project will allow users to adjust some training hyperparameters, visualize the training progress, and test the trained model by providing their own inputted images/values. The project is aptly named JTorch, to combine the idea of **J**ava and Py**Torch**, the well-known but python-based deep learning framework.

### Who will use this application?

The scope of this project is limited and likely won't embrace the many aspects of high-class frameworks like PyTorch, such as GPU acceleration or optimizations for massive datasets. With this in mind, the audience for this application are **students who want to tinker with Neural Networks in a graphical manner**. Being able to alter different training parameters, visualize the learning process of a Neural Network, and testing it directly, are all things that would be educational and valuable to students who want to learn more about Deep Learning.

### Why is this project of interest to me?

I am an aspiring Machine Learning Engineer and Research Scientist, with growing experience in building models that shape how machines interpret language, vision, and data all around us. I'm passionate about teaching difficult concepts in Deep Learning to people around me, and through this application, I myself gain a refresher on Deep Learning under the hood, and the opportunity to showcase this in an educational manner to students and colleagues alike.

## User Stories

- As a user, I want to create tensors and perform basic operations like addition and multiplication to prepare data for my neural network.

- As a user, I want to add layers (like Dense or Activation layers) to my neural network model so I can customize its architecture.

- As a user, I want to train the neural network on a dataset to develop a model that can make accurate predictions.

- As a user, I want to monitor training metrics such as loss and accuracy to evaluate the model's performance over time.

- As a user, I want to choose the optimization algorithm (e.g., SGD) for training my neural network to improve learning efficiency.

- As a user, I want to view the neural network’s architecture, including the layers and their connections, to understand the structure of my model.

- As a user, I want to have the option to save my neural network's architecture, weights, and biases after training.

- As a user, I want to be able to load my neural network's architecture, weights, and biases.

## Instructions for End User

- **How to add layers to the neural network:**

  You can add layers (X) to the neural network (Y) by:

  1. Navigate to the **"Add Layers"** tab in the application.
  2. Click the **"Add Layer"** button.
  3. In the dialog that appears, select the layer type you want to add (e.g., **"Dense Layer"** or **"Activation Layer"**).
  4. Fill in the required parameters for the layer:
     - For **Dense Layer**, enter the **Input Size** and **Output Size**.
     - For **Activation Layer**, enter the **Activation Function**.
  5. Click **"OK"** to add the layer to the neural network.
  6. The layer will appear in the list of layers, showing its details.

- **You can generate the first required action related to the user story "adding multiple layers to the neural network" by editing a layer:**

  To **edit** a layer:

  1. In the **"Add Layers"** tab, locate the layer you wish to edit in the list.
  2. Click the **"Edit"** button next to that layer.
  3. A dialog will open with the layer's current parameters.
  4. Modify the parameters as needed.
  5. Click **"OK"** to save the changes.
  6. The updated layer will reflect the new parameters in the list.

- **You can generate the second required action related to the user story "adding multiple layers to the neural network" by deleting a layer:**

  To **delete** a layer:

  1. In the **"Add Layers"** tab, find the layer you wish to delete in the list.
  2. Click the **"Delete"** button next to that layer.
  3. Confirm the deletion when prompted.
  4. The layer will be removed from the neural network and the list.

- **You can locate my visual component by navigating to the "Loss Graph" tab:**

  After training the neural network, the **"Loss Graph"** tab displays a visual graph of the training loss over epochs. This graph allows you to monitor training metrics and evaluate the model's performance over time.

- **You can save the state of my application by:**

  - **Saving the Neural Network:**

    1. In the **"Add Layers"** tab, after adding layers to your network, click the **"Save Network"** button.
    2. Choose a location and filename in the file dialog to save the neural network to a file.

  - **Saving Tensors:**

    1. In the **"Add Tensors"** tab, select a tensor from the list.
    2. Click the **"Save Tensor"** button.
    3. Choose a location and filename in the file dialog to save the tensor.

  - **Saving the Optimizer:**

    1. In the **"Train Network"** tab, after configuring the optimizer settings, click the **"Save Optimizer"** button.
    2. Choose a location and filename in the file dialog to save the optimizer settings.

- **You can reload the state of my application by:**

  - **Loading the Neural Network:**

    1. In the **"Add Layers"** tab, click the **"Load Network"** button.
    2. Select the neural network file you wish to load from the file dialog.
    3. The layers will be loaded into the application and displayed in the list.

  - **Loading Tensors:**

    1. In the **"Add Tensors"** tab, click the **"Load Tensor"** button.
    2. Select the tensor file you wish to load from the file dialog.
    3. Enter a unique name for the loaded tensor when prompted.
    4. The tensor will be added to the list of tensors in the application.

  - **Loading the Optimizer:**

    1. In the **"Train Network"** tab, click the **"Load Optimizer"** button.
    2. Select the optimizer file you wish to load from the file dialog.
    3. The optimizer settings will be loaded into the application fields.

## Phase 4: Task 2

### Representative sample of events

```
    Event Log:
    Thu Nov 28 22:33:54 PST 2024
    Dense layer initialized with input size 3 and output size 2
    Thu Nov 28 22:33:54 PST 2024
    Added layer: Dense Layer (3 -> 2) to the network. Network now has 1 layers.
    Thu Nov 28 22:33:59 PST 2024
    Dense layer initialized with input size 3 and output size 3
    Thu Nov 28 22:33:59 PST 2024
    Updated layer at index 0 with new layer: Dense Layer (3 -> 3)
    Thu Nov 28 22:34:06 PST 2024
    Dense layer initialized with input size 3 and output size 2
    Thu Nov 28 22:34:06 PST 2024
    Updated layer at index 0 with new layer: Dense Layer (3 -> 2)
    Thu Nov 28 22:34:35 PST 2024
    Deserialized Tensor from JSON with dimensions 2x3
    Thu Nov 28 22:34:35 PST 2024
    Deserialized Tensor from JSON with dimensions 1x3
    Thu Nov 28 22:34:35 PST 2024
    Deserialized DenseLayer from JSON with input size 2 and output size 3
    Thu Nov 28 22:34:35 PST 2024
    Dense layer initialized with preloaded weights and biases. Dense Layer (2 -> 3)
    Thu Nov 28 22:34:35 PST 2024
    Added layer: Dense Layer (2 -> 3) to the network. Network now has 1 layers.
    Thu Nov 28 22:34:35 PST 2024
    Deserialized ActivationLayer from JSON with function 'sigmoid'
    Thu Nov 28 22:34:35 PST 2024
    Initialized ActivationLayer with function 'sigmoid'
    Thu Nov 28 22:34:35 PST 2024
    Added layer: Activation Layer (sigmoid) to the network. Network now has 2 layers.
    Thu Nov 28 22:34:35 PST 2024
    Deserialized Tensor from JSON with dimensions 3x1
    Thu Nov 28 22:34:35 PST 2024
    Deserialized Tensor from JSON with dimensions 1x1
    Thu Nov 28 22:34:35 PST 2024
    Deserialized DenseLayer from JSON with input size 3 and output size 1
    Thu Nov 28 22:34:35 PST 2024
    Dense layer initialized with preloaded weights and biases. Dense Layer (3 -> 1)
    Thu Nov 28 22:34:35 PST 2024
    Added layer: Dense Layer (3 -> 1) to the network. Network now has 3 layers.
    Thu Nov 28 22:34:35 PST 2024
    Deserialized ActivationLayer from JSON with function 'sigmoid'
    Thu Nov 28 22:34:35 PST 2024
    Initialized ActivationLayer with function 'sigmoid'
    Thu Nov 28 22:34:35 PST 2024
    Added layer: Activation Layer (sigmoid) to the network. Network now has 4 layers.
    Thu Nov 28 22:34:35 PST 2024
    Deserialized NeuralNetwork from JSON with 4 layers.
    Thu Nov 28 22:34:52 PST 2024
    Dense layer initialized with input size 3 and output size 2
    Thu Nov 28 22:34:52 PST 2024
    Updated layer at index 2 with new layer: Dense Layer (3 -> 2)
    Thu Nov 28 22:34:57 PST 2024
    Dense layer initialized with input size 3 and output size 1
    Thu Nov 28 22:34:57 PST 2024
    Updated layer at index 2 with new layer: Dense Layer (3 -> 1)
    Thu Nov 28 22:35:17 PST 2024
    Deserialized Tensor from JSON with dimensions 4x2
    Thu Nov 28 22:35:27 PST 2024
    Deserialized Tensor from JSON with dimensions 4x1
    Thu Nov 28 22:35:42 PST 2024
    Initialized SgdOptimizer with learning rate: 0.01
    Thu Nov 28 22:35:42 PST 2024
    Training started for 5 epochs with optimizer: SgdOptimizer
    Thu Nov 28 22:35:42 PST 2024
    Epoch 1/5 completed. Average Loss: 0.2514277657226767
    Thu Nov 28 22:35:42 PST 2024
    Epoch 2/5 completed. Average Loss: 0.2514174612400747
    Thu Nov 28 22:35:42 PST 2024
    Epoch 3/5 completed. Average Loss: 0.25140723775665985
    Thu Nov 28 22:35:42 PST 2024
    Epoch 4/5 completed. Average Loss: 0.2513970946325078
    Thu Nov 28 22:35:42 PST 2024
    Epoch 5/5 completed. Average Loss: 0.25138703123260453
    Thu Nov 28 22:35:42 PST 2024
    Training completed after 5 epochs.
```