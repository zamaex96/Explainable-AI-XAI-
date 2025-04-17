
**Purpose and Goal**

The primary purpose of this script is to build, train, evaluate, and *explain* a hybrid deep learning model for a classification task. The model combines Convolutional Neural Networks (CNNs), Long Short-Term Memory networks (LSTMs), and an Attention mechanism.

The key goals are:

1.  **Classification:** Train a model to classify input data (presumably sequences or feature vectors) into one of `OUTPUT_SIZE` categories.
2.  **Hybrid Architecture:** Leverage the strengths of different neural network types:
    * **CNNs:** To automatically extract spatial or local features from the input sequence (treating the features as a 1D sequence).
    * **LSTMs:** To model temporal dependencies or sequential patterns in the features extracted by the CNN (using a bidirectional LSTM captures patterns in both forward and backward directions).
    * **Attention:** To allow the model to focus on the most relevant parts of the sequence processed by the LSTM when making the final classification decision.
3.  **Explainability (XAI):** Go beyond just prediction accuracy and understand *why* the model makes certain predictions. This is achieved using:
    * **SHAP (SHapley Additive exPlanations):** To quantify the contribution of each input feature to the model's output for specific predictions and globally.
    * **Grad-CAM (Gradient-weighted Class Activation Mapping):** To visualize which parts of the input sequence were most important for the *CNN layers* when classifying a specific sample.
    * **Attention Weights Visualization:** To visualize the internal focus of the *Attention layer* within the sequence processed by the LSTM.
4.  **Reproducibility & Reusability:** Save the trained model (weights and hyperparameters) so it can be loaded and used later without retraining.
5.  **Performance Monitoring:** Track and visualize training and testing loss/accuracy over epochs to monitor the learning process.

**Step-by-Step Implementation Process**

Here's a breakdown of the script's sections with code snippets and explanations:

1.  **Imports:**
    * Standard libraries like `torch`, `torch.nn`, `torch.optim`, `DataLoader`, `matplotlib`, `pandas`, `numpy`, `seaborn`, `random` are imported for deep learning, data handling, plotting, and utility functions.
    * `shap` is imported for explainability analysis.
    * `sklearn.metrics` is used for evaluation metrics like confusion matrix and classification report.

    ```python
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Subset
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import shap
    from sklearn.metrics import confusion_matrix, classification_report
    import seaborn as sns
    import random
    import os # Added for checking file existence in loading part

    # Assuming data_loader.py contains CustomDataset
    from data_loader import CustomDataset # Adjust import as needed
    ```

2.  **Custom Dataset Assumption:**
    * The script assumes you have a `CustomDataset` class (likely defined in `data_loader.py`) that inherits from `torch.utils.data.Dataset`.
    * This class is responsible for loading data (e.g., from a CSV), extracting features and labels, and returning them as tensors.
    * **Crucially**, for classification with `nn.CrossEntropyLoss`, the `__getitem__` method should return `(data_tensor, label_tensor)`, where `label_tensor` is a single integer (scalar tensor of type `torch.long`) representing the class index (0, 1, 2,...).
    * The example provided shows how such a dataset might look, including optional attributes `feature_names` and `class_names` which are very useful for interpreting the explainability results.

3.  **Configuration:**
    * Sets up all the important parameters and constants for the experiment.
    * `INPUT_SIZE`: Number of features in each input sample.
    * `CNN_CHANNELS`, `LSTM_HIDDEN_SIZE`, `LSTM_NUM_LAYERS`, `NUM_HEADS_ATTENTION`: Hyperparameters defining the model architecture.
    * `OUTPUT_SIZE`: Number of distinct classes the model should predict.
    * `LEARNING_RATE`, `BATCH_SIZE`, `EPOCHS`: Training parameters.
    * `DEVICE`: Automatically selects GPU (`cuda`) if available, otherwise CPU.
    * `TRAIN_CSV_PATH`, `TEST_CSV_PATH`: Paths to the data files.
    * `SAVED_MODEL_PATH`, `RESULTS_CSV_PATH`, `PLOT_SAVE_PATH`: Paths for saving outputs.
    * `NUM_SHAP_BACKGROUND_SAMPLES`, `NUM_SAMPLES_TO_EXPLAIN`: Parameters for the explainability analysis.
    * It attempts to load `FEATURE_NAMES` and `CLASS_NAMES` from the dataset, falling back to generic names if the attributes don't exist. This is good practice for labeling plots and outputs later.

    ```python
    # --- Configuration ---
    INPUT_SIZE = 4
    CNN_CHANNELS = 16
    LSTM_HIDDEN_SIZE = 32
    LSTM_NUM_LAYERS = 2
    OUTPUT_SIZE = 4
    NUM_HEADS_ATTENTION = 4
    LEARNING_RATE = 0.005
    BATCH_SIZE = 32
    EPOCHS = 50
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TRAIN_CSV_PATH = "train_set.csv"
    TEST_CSV_PATH = "test_set.csv"
    SAVED_MODEL_PATH = 'saved_model_hybrid_explained.pth'
    RESULTS_CSV_PATH = "train_results_hybrid_explained.csv"
    PLOT_SAVE_PATH = "train_plots_hybrid_explained.eps"
    NUM_SHAP_BACKGROUND_SAMPLES = 100
    NUM_SAMPLES_TO_EXPLAIN = 5

    # Define feature and class names
    try:
        temp_dataset = CustomDataset(TRAIN_CSV_PATH)
        FEATURE_NAMES = temp_dataset.feature_names
        CLASS_NAMES = temp_dataset.class_names
        # ... (rest of try block)
    except AttributeError:
        # ... (fallback names)
    ```

4.  **Model Definition (`HybridCNNLSTMAttention`):**
    * Defines the neural network architecture inheriting from `nn.Module`.
    * `__init__`:
        * Initializes CNN layers (`nn.Conv1d`, `nn.ReLU`, `nn.BatchNorm1d`, `nn.MaxPool1d`). BatchNorm is added for better stability and faster training. MaxPool reduces the sequence length.
        * *Dynamically calculates the sequence length* output by the CNN (`cnn_output_seq_len`) based on `INPUT_SIZE` and pooling layers. This is important for defining the subsequent LSTM layer correctly.
        * Initializes a bidirectional LSTM layer (`nn.LSTM`). `batch_first=True` makes tensor handling easier. `bidirectional=True` doubles the effective hidden size.
        * Initializes a `nn.MultiheadAttention` layer. It takes the LSTM outputs as query, key, and value (self-attention).
        * Initializes the final fully connected layer (`nn.Linear`) to map the processed sequence representation to the `OUTPUT_SIZE` logits.
        * Initializes `feature_maps` and `gradients` to `None` and defines `activations_hook`. These are specifically for capturing intermediate results needed for Grad-CAM.
    * `forward(self, x)`:
        * Defines the data flow through the layers.
        * Input `x` shape: `[batch_size, input_size]`.
        * `x.unsqueeze(1)`: Reshapes input for `Conv1d` to `[batch_size, 1, input_size]`.
        * Passes through CNN layers.
        * **Grad-CAM Hook:** Registers the `activations_hook` to the *output* of the last CNN layer's activation (`x_cnn`). This hook will store the gradients flowing back into this layer during `backward()`. The feature map itself is also stored. This is only done during evaluation (`self.training is False`) if gradients are needed.
        * `x_cnn.permute(0, 2, 1)`: Reshapes the CNN output to `[batch_size, seq_len_after_cnn, cnn_channels*2]` suitable for the LSTM layer (`batch_first=True`).
        * Passes through the bidirectional LSTM.
        * Passes the LSTM output sequence through the `MultiheadAttention` layer.
        * **Context Aggregation:** Aggregates the attention output sequence (`attn_output`) into a single vector (`context_vector`) per sample using mean pooling (`torch.mean(attn_output, dim=1)`). This vector summarizes the sequence information weighted by attention. Other options like using the last hidden state or max pooling are commented out.
        * Passes the `context_vector` through the final `nn.Linear` layer to get classification logits.
        * Returns the final `output` logits and the `attn_weights` (useful for visualization).

    ```python
    class HybridCNNLSTMAttention(nn.Module):
        def __init__(self, input_size, cnn_channels, lstm_hidden_size, lstm_num_layers, output_size, num_heads):
            super(HybridCNNLSTMAttention, self).__init__()
            # ... (CNN layers definition) ...
            self.cnn_output_seq_len = input_size // 4
            lstm_input_features = cnn_channels * 2
            # ... (LSTM layer definition) ...
            lstm_output_size = lstm_hidden_size * 2
            # ... (Attention layer definition) ...
            self.fc = nn.Linear(lstm_output_size, output_size)
            # ... (Grad-CAM hooks init) ...

        def activations_hook(self, grad):
            self.gradients = grad

        def forward(self, x):
            # ... (CNN forward pass) ...
            if self.training is False and x_cnn.requires_grad:
                h = x_cnn.register_hook(self.activations_hook)
                self.feature_maps = x_cnn
            # ... (LSTM forward pass) ...
            # ... (Attention forward pass) ...
            context_vector = torch.mean(attn_output, dim=1) # Aggregate attention output
            output = self.fc(context_vector)
            return output, attn_weights
    ```

5.  **Training Setup:**
    * Instantiates the `HybridCNNLSTMAttention` model.
    * Moves the model to the selected `DEVICE`.
    * Defines the loss function: `nn.CrossEntropyLoss` combines LogSoftmax and Negative Log-Likelihood loss, suitable for multi-class classification.
    * Defines the optimizer: `optim.Adam` is a popular adaptive learning rate optimizer.
    * Defines a learning rate scheduler: `optim.lr_scheduler.StepLR` reduces the learning rate by a factor (`gamma=0.1`) every `step_size` epochs (here, every third of the total epochs). This helps in fine-tuning the model in later stages.

    ```python
    model = HybridCNNLSTMAttention(...)
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=EPOCHS // 3, gamma=0.1)
    ```

6.  **Data Loading:**
    * Creates instances of the `CustomDataset` for training and testing data using the specified CSV paths.
    * Creates `DataLoader` instances, which handle batching, shuffling (for training data), and parallel data loading.

    ```python
    train_dataset = CustomDataset(TRAIN_CSV_PATH)
    test_dataset = CustomDataset(TEST_CSV_PATH)
    train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    ```

7.  **Training and Testing Loop:**
    * Initializes lists to store loss and accuracy values for plotting.
    * Iterates through the specified number of `EPOCHS`.
    * **Training Phase:**
        * Sets model to training mode (`model.train()`). This enables dropout, batch normalization updates, etc.
        * Iterates through batches from `train_data_loader`.
        * Moves input `inputs` and `labels` to the `DEVICE`.
        * Clears previous gradients (`optimizer.zero_grad()`).
        * Performs a forward pass: `outputs, _ = model(inputs)`. The attention weights are ignored for loss calculation.
        * Calculates the loss using `criterion(outputs, labels)`.
        * Performs backpropagation (`loss.backward()`) to compute gradients.
        * Updates model weights (`optimizer.step()`).
        * Accumulates loss and calculates the number of correct predictions for the epoch.
        * Calculates and stores average training loss and accuracy for the epoch.
    * **Testing Phase:**
        * Sets model to evaluation mode (`model.eval()`). This disables dropout and uses running stats for batch normalization.
        * Disables gradient calculation (`with torch.no_grad():`) for efficiency and to prevent accidental updates.
        * Iterates through batches from `test_data_loader`.
        * Performs a forward pass.
        * Calculates the loss.
        * Calculates the number of correct predictions.
        * **Stores all predictions (`all_preds`) and true labels (`all_labels`)** from the test set. These are needed later for the confusion matrix and classification report.
        * Calculates and stores average test loss and accuracy for the epoch.
    * Updates the learning rate using `scheduler.step()`.
    * Prints the progress (loss, accuracy, learning rate) periodically.

    ```python
    train_loss_values = []
    # ... (other lists init) ...

    for epoch in range(EPOCHS):
        # --- Training Phase ---
        model.train()
        # ... (training loop over batches) ...
        train_loss_values.append(epoch_train_loss)
        train_accuracy_values.append(train_accuracy)

        # --- Testing Phase ---
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            # ... (testing loop over batches) ...
            all_preds.extend(predicted_test.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        # ... (store test metrics) ...
        scheduler.step()
        # ... (print progress) ...
    ```

8.  **Save Model:**
    * Creates a `hyperparameters` dictionary containing the key architectural parameters used to build the model. Also includes feature/class names.
    * Creates a `checkpoint` dictionary containing:
        * The `hyperparameters`.
        * The learned model weights (`model.state_dict()`).
        * The last epoch number.
        * Optionally, the state of the optimizer and scheduler (useful if you want to resume training later).
    * Saves this `checkpoint` dictionary to the specified `.pth` file using `torch.save()`. This approach is better than saving the entire model object, as it's more portable across different code versions.

    ```python
    print(f"\nSaving model checkpoint to {SAVED_MODEL_PATH}...")
    hyperparameters = { ... } # Populate with config values
    checkpoint = {
        'hyperparameters': hyperparameters,
        'model_state_dict': model.state_dict(),
        # ... (optional optimizer/scheduler states) ...
    }
    torch.save(checkpoint, SAVED_MODEL_PATH)
    ```

9.  **Save Training Results:**
    * Creates a pandas DataFrame storing the epoch number, train/test loss, and train/test accuracy for each epoch.
    * Saves this DataFrame to a CSV file.

    ```python
    train_info_df = pd.DataFrame({
        'epoch': range(1, EPOCHS + 1),
        # ... (other columns) ...
    })
    train_info_df.to_csv(RESULTS_CSV_PATH, index=False)
    ```

10. **Plotting:**
    * Uses `matplotlib.pyplot` to create two subplots:
        * Loss vs. Epochs (Training and Testing)
        * Accuracy vs. Epochs (Training and Testing)
    * Saves the combined plot to an EPS file (vector format) and displays it.

    ```python
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1) # Loss plot
    # ... (plotting commands) ...
    plt.subplot(1, 2, 2) # Accuracy plot
    # ... (plotting commands) ...
    plt.tight_layout()
    plt.savefig(PLOT_SAVE_PATH, format='eps')
    plt.show()
    ```

11. **Explainability Analysis:** This is a key part demonstrating XAI techniques.
    * Ensures the model is in evaluation mode (`model.eval()`).
    * **a. Confusion Matrix & Classification Report:**
        * Uses the `all_labels` and `all_preds` collected during testing.
        * Calculates the confusion matrix using `sklearn.metrics.confusion_matrix`.
        * Visualizes the matrix using `seaborn.heatmap`.
        * Prints the `sklearn.metrics.classification_report`, which provides precision, recall, F1-score, and support for each class.
    * **b. SHAP Analysis:**
        * **Background Data:** Selects a random subset of the *training* data to serve as a background distribution for SHAP's `DeepExplainer`. This background helps establish a baseline expectation.
        * **Wrapper Function:** Defines `model_predictor`. This is crucial because the model's `forward` method returns `(output, attn_weights)`, but SHAP explainers typically expect a function that takes a tensor and returns *only the model's output tensor* (logits or probabilities).
        * **Explainer:** Creates a `shap.DeepExplainer` instance, passing it the `model_predictor` and the `background_data`.
        * **Calculate SHAP Values:** Selects a subset of *test* samples and calculates their SHAP values using `explainer.shap_values(test_samples)`. This can be computationally intensive. The result `shap_values` is typically a list (one element per output class) of arrays, where each array has shape `[num_samples, num_features]`.
        * **Summary Plots:**
            * Generates a bar plot (`plot_type="bar"`) showing the mean absolute SHAP value for each feature across all samples and classes (global importance).
            * Generates dot summary plots (`plot_type="dot"`, the default) for each class individually, showing the distribution of SHAP values for each feature and how high/low feature values impact the prediction for that specific class.
        * **Force Plots:**
            * For a few individual test samples:
                * Gets the model's prediction and true label.
                * Uses `shap.force_plot` to visualize how each feature contributed to pushing the model's output away from the base value (average prediction over the background dataset, stored in `explainer.expected_value`) towards the actual prediction *for the predicted class*.
                * Optionally, plots the force plot for the *true class* if it was different from the predicted one.
    * **c. Grad-CAM:**
        * **`get_grad_cam` function:**
            * Takes the model, the target layer's output (feature maps captured by the hook), the target class index, and the final model output.
            * Performs backpropagation specifically for the `target_class_index` score (`output[:, target_class_index].backward(retain_graph=True)`). `retain_graph=True` might be needed if backpropagating multiple times.
            * Retrieves the gradients stored by the hook (`model.gradients`).
            * Calculates the importance weights (alpha) by averaging gradients across the sequence dimension (`torch.mean(gradients, dim=[2])`).
            * Computes the weighted sum of feature map channels using these weights.
            * Applies ReLU (`F.relu`) to keep only positive contributions.
            * Normalizes the resulting heatmap for visualization.
        * **Visualization Loop:**
            * Selects a few test samples.
            * For each sample:
                * Ensures the input tensor requires gradients (`input_tensor.requires_grad = True`).
                * Performs a forward pass to get the prediction *and* trigger the activation/gradient hooks (storing `model.feature_maps` and enabling `model.gradients` upon backward).
                * Calls `get_grad_cam` to compute the heatmap for the predicted class.
                * **Upsamples** the heatmap (which has length `cnn_output_seq_len`) to the original `INPUT_SIZE` using `F.interpolate` (linear interpolation) because the heatmap corresponds to the CNN's output sequence, but we want to visualize it relative to the input features.
                * Plots the original input features and overlays the upsampled heatmap using `plt.imshow` with transparency (`alpha`).
                * Cleans up gradients/hooks (good practice).
    * **d. Attention Weights Visualization:**
        * Selects a few test samples.
        * Performs a forward pass *with no gradients* (`torch.no_grad()`) to get the `attn_weights` returned by the model.
        * Extracts and potentially processes the attention weights (e.g., squeezing batch dimension, averaging over heads if it's MultiheadAttention). The shape depends on the specific Attention layer implementation. The example assumes `[batch, seq_len, seq_len]` after processing/squeezing.
        * Uses `seaborn.heatmap` to visualize the attention matrix, showing which parts of the sequence (Query Time Steps) attended to which other parts (Key Time Steps) *at the output of the LSTM*.

12. **Load Model Example:**
    * Demonstrates how to reload the saved model.
    * Checks if the `SAVED_MODEL_PATH` exists.
    * Loads the `checkpoint` dictionary using `torch.load`. `map_location=DEVICE` ensures the model is loaded onto the correct device, regardless of where it was saved.
    * Extracts the `hyperparameters` from the checkpoint.
    * **Re-instantiates the model** using these loaded hyperparameters. This is crucial for ensuring the architecture matches the saved weights.
    * Loads the learned weights using `loaded_model.load_state_dict(checkpoint['model_state_dict'])`.
    * Moves the loaded model to the `DEVICE`.
    * **Sets the model to evaluation mode (`loaded_model.eval()`)** immediately after loading, as it's typically used for inference.
    * Includes an example of using the loaded model to make a prediction on a sample from the test set and prints the true vs. predicted class.

    ```python
    print("\n--- Loading Model Example ---")
    if os.path.exists(SAVED_MODEL_PATH):
        checkpoint = torch.load(SAVED_MODEL_PATH, map_location=DEVICE)
        loaded_hyperparameters = checkpoint['hyperparameters']
        loaded_model = HybridCNNLSTMAttention(
            input_size=loaded_hyperparameters['input_size'],
            # ... (rest of params from loaded_hyperparameters) ...
        )
        loaded_model.load_state_dict(checkpoint['model_state_dict'])
        loaded_model.to(DEVICE)
        loaded_model.eval()
        print("Model loaded successfully.")
        # ... (Test prediction with loaded model) ...
    else:
        print(f"Error: Saved model file not found at {SAVED_MODEL_PATH}. Cannot load.")

    ```

This comprehensive script provides a solid template for building, training, and thoroughly analyzing a hybrid deep learning model, placing a strong emphasis on understanding the model's behavior through various explainability techniques.

<div align="center">
  <a href="https://maazsalman.org/">
    <img width="70" src="click-svgrepo-com.svg" alt="gh" />
  </a>
  <p> Explore More! 🚀</p>
</div>
