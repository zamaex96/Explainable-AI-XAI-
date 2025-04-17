
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

Okay, let's delve deeper into the three explainability techniques used in the script: SHAP, Grad-CAM, and Attention Weights Visualization.

---

### 1. SHAP (SHapley Additive exPlanations) Analysis

* **Purpose:** SHAP aims to explain the output of *any* machine learning model by assigning an importance value (the "SHAP value") to each **input feature** for a particular prediction. It tells you how much each feature contributed, positively or negatively, to pushing the model's output away from a baseline (average) prediction towards the final prediction for that specific instance. It's based on Shapley values from cooperative game theory, providing a theoretically sound way to distribute the prediction "payout" among the features.
* **Concept (for DeepExplainer):** Calculating exact Shapley values is computationally infeasible for complex models. `shap.DeepExplainer` (used here as it's suitable for neural networks) provides an efficient approximation. It combines ideas from DeepLIFT (another attribution method using backpropagation) and Shapley values. It requires a **background dataset** to represent the expected distribution of features, which helps estimate the baseline prediction (the "expected value" or `explainer.expected_value`). The SHAP values are calculated by comparing the model's output when a feature is present versus when it's "absent" (approximated using the background data), considering all possible feature orderings or subsets (approximated efficiently).
* **Step-by-Step Implementation (from the script):**

    1.  **Select Background Data:**
        * **Code:**
            ```python
            background_indices = random.sample(range(len(train_dataset)), min(NUM_SHAP_BACKGROUND_SAMPLES, len(train_dataset)))
            background_data = torch.stack([train_dataset[i][0] for i in background_indices]).to(DEVICE)
            ```
        * **Explanation:** A subset of the *training data* is chosen randomly. This data acts as the reference or baseline distribution. `DeepExplainer` uses this to understand the "average" behavior of the model and to simulate the effect of features being "absent" or unobserved when calculating contributions. The number of samples (`NUM_SHAP_BACKGROUND_SAMPLES`) is a trade-off between representativeness and computational cost. The data is stacked into a single tensor and moved to the correct device.

    2.  **Define Model Predictor Wrapper:**
        * **Code:**
            ```python
            def model_predictor(data_tensor):
                output, _ = model(data_tensor)
                return output
            ```
        * **Explanation:** The `HybridCNNLSTMAttention` model's `forward` method returns `(output, attn_weights)`. However, the SHAP explainer needs a function that takes the input tensor and returns *only* the tensor representing the model's predictions (logits in this case). This wrapper function serves exactly that purpose, discarding the attention weights.

    3.  **Instantiate Explainer:**
        * **Code:**
            ```python
            explainer = shap.DeepExplainer(model_predictor, background_data)
            ```
        * **Explanation:** Creates the `DeepExplainer` object, providing it with the wrapped model function (`model_predictor`) and the prepared background data tensor.

    4.  **Select Test Samples to Explain:**
        * **Code:**
            ```python
            test_indices = random.sample(range(len(test_dataset)), min(NUM_SAMPLES_TO_EXPLAIN * 5, len(test_dataset))) # Get more samples for summary
            test_samples = torch.stack([test_dataset[i][0] for i in test_indices]).to(DEVICE)
            # ... (get corresponding labels if needed) ...
            ```
        * **Explanation:** A subset of the *test data* is selected for explanation. More samples might be selected than strictly needed for individual force plots (`NUM_SAMPLES_TO_EXPLAIN`) to provide a better basis for the summary plots.

    5.  **Calculate SHAP Values:**
        * **Code:**
            ```python
            shap_values = explainer.shap_values(test_samples)
            ```
        * **Explanation:** This is the core computation. The explainer calculates SHAP values for each feature, for each selected test sample, and for each output class. The output `shap_values` is typically a list, where `shap_values[i]` is a NumPy array containing the SHAP values for the i-th class. Each array has the shape `(num_test_samples, num_features)`.

    6.  **Generate Summary Plots:**
        * **Code (Bar Plot - Global Importance):**
            ```python
            shap.summary_plot(shap_values, test_samples.cpu().numpy(), feature_names=FEATURE_NAMES,
                              class_names=CLASS_NAMES, plot_type="bar", max_display=10)
            # ... (title, save, show) ...
            ```
        * **Explanation (Bar Plot):** Aggregates importance across all classes and samples by taking the mean absolute SHAP value for each feature. The resulting bar chart shows the top `max_display` features ranked by their overall impact on the model's predictions.
        * **Code (Dot Plot - Per-Class Importance):**
            ```python
            for i, class_name in enumerate(CLASS_NAMES):
                plt.figure()
                shap.summary_plot(shap_values[i], test_samples.cpu().numpy(), feature_names=FEATURE_NAMES,
                                  show=False, max_display=10)
                # ... (title, save, close) ...
            ```
        * **Explanation (Dot Plot):** This plot is generated *per class*. For each feature (y-axis), it plots a point for each sample. The point's position on the x-axis represents the SHAP value (impact on that class's prediction), and its color represents the original value of that feature (high/low). This reveals not just *which* features are important for a class, but also *how* their values influence the prediction (e.g., high values of Feature_3 tend to increase the prediction score for Class_1).

    7.  **Generate Force Plots (Individual Explanations):**
        * **Code:**
            ```python
            # ... (get base_values = explainer.expected_value) ...
            # ... (loop through selected indices_to_plot) ...
            if base_values is not None and len(base_values) == OUTPUT_SIZE:
                shap.force_plot(base_values[predicted_label], # Base value for the predicted class
                                shap_values[predicted_label][i,:], # SHAP values for this sample & predicted class
                                test_samples_np[i,:], # Feature values for this sample
                                feature_names=FEATURE_NAMES,
                                matplotlib=True, show=False)
                # ... (title, save, close) ...
            # ... (optional: plot for true_label if different) ...
            ```
        * **Explanation:** For a single prediction:
            * `base_value`: The average prediction score for that class across the background dataset (`explainer.expected_value[predicted_label]`).
            * `shap_values[...]`: The SHAP values calculated for this specific sample (`i`) and the class of interest (`predicted_label`).
            * `test_samples_np[i,:]`: The actual feature values for this sample.
            * The plot visualizes features as forces pushing the output from the `base_value` towards the final prediction score. Red features push the score higher (increase probability of that class), blue features push it lower. The size of the feature's block indicates the magnitude of its impact. `matplotlib=True` allows saving the plot. Plotting for both the predicted and true (if different) classes helps understand misclassifications.

---

### 2. Grad-CAM (Gradient-weighted Class Activation Mapping)

* **Purpose:** Grad-CAM is designed to produce a coarse localization map highlighting the important regions *in the input space or feature maps of a CNN* that the model used to predict a specific target class. It answers: "Which parts of the sequence (as seen by the target CNN layer) were most important for deciding on class Y?". It's specific to Convolutional Neural Networks.
* **Concept:** It leverages the spatial information preserved in the feature maps of convolutional layers.
    1.  It calculates the gradient of the score for the target class with respect to the feature maps of a chosen (usually the last) convolutional layer. These gradients indicate how much a change in each feature map channel affects the class score.
    2.  It computes the average of these gradients across the spatial dimensions (global average pooling) for each channel. This gives a weight (importance score, `alpha_k`) for each feature map channel.
    3.  It computes a weighted combination of the forward activation feature maps, using the channel importance weights (`alpha_k`) derived from the gradients.
    4.  It applies a ReLU function to this combination. This focuses on features that have a *positive* influence on the class of interest. The result is a heatmap indicating regions the CNN focused on for that class prediction.
* **Step-by-Step Implementation (from the script):**

    1.  **Model Hooks Setup:**
        * **Code (in `__init__`):**
            ```python
            self.feature_maps = None
            self.gradients = None
            def activations_hook(self, grad):
                self.gradients = grad
            ```
        * **Code (in `forward`):**
            ```python
            # Hook applied to the output of the last CNN activation
            if self.training is False and x_cnn.requires_grad:
                h = x_cnn.register_hook(self.activations_hook) # Register hook to capture gradient
                self.feature_maps = x_cnn # Store the activation map itself
            ```
        * **Explanation:** The `activations_hook` function is defined to simply store the gradient that flows back into the layer it's attached to. In the `forward` pass (only during evaluation when gradients might be needed for explanation), this hook is registered to the tensor `x_cnn` (output of the last CNN layer block). The activation tensor `x_cnn` itself is also stored in `self.feature_maps`. When `loss.backward()` (or specifically `output[:, target_class_index].backward()`) is called later, PyTorch calculates gradients, and the hook automatically saves the gradient flowing into `x_cnn` into `self.gradients`.

    2.  **`get_grad_cam` Function:**
        * **Code:**
            ```python
            def get_grad_cam(model, target_layer_output, target_class_index, output):
                if model.gradients is None or target_layer_output is None: return None # Check if hooks worked
                # Backpropagate the specific class score:
                output[:, target_class_index].backward(retain_graph=True)
                gradients = model.gradients # Shape: [batch, channels, seq_len]
                activations = target_layer_output # Shape: [batch, channels, seq_len]

                # Pool gradients (alpha_k calculation):
                pooled_gradients = torch.mean(gradients, dim=[2]) # Shape: [batch, channels]

                # Weight the channels (importance weighting):
                pooled_gradients = pooled_gradients.unsqueeze(-1) # Shape: [batch, channels, 1] for broadcasting
                heatmap = torch.sum(activations * pooled_gradients, dim=1) # Shape: [batch, seq_len]
                heatmap = F.relu(heatmap) # Apply ReLU

                # Normalize:
                # ... (normalization code) ...
                return heatmap.squeeze().cpu().numpy()
            ```
        * **Explanation:**
            * Takes the model, the captured feature maps (`target_layer_output` which is `model.feature_maps`), the index of the class to explain, and the model's final output tensor.
            * `output[:, target_class_index].backward(retain_graph=True)`: This is the key step to get gradients *relevant to the target class*. It backpropagates *only* from the score of the desired class. `retain_graph=True` might be needed if you perform multiple backward passes (e.g., explaining multiple classes for the same input).
            * Retrieves `gradients` and `activations` stored by the hooks.
            * `torch.mean(gradients, dim=[2])`: Calculates the importance weight (`alpha_k`) for each channel by averaging the gradient values across the sequence dimension (dimension 2).
            * `activations * pooled_gradients`: Weights the activation maps channel-wise using the importance weights (broadcasting handles the dimensions).
            * `torch.sum(..., dim=1)`: Sums the weighted channels to create the raw heatmap (collapsing the channel dimension).
            * `F.relu(heatmap)`: Keeps only positive contributions.
            * Normalization: Scales the heatmap values to be between 0 and 1 for easier visualization.
            * Returns the heatmap as a NumPy array.

    3.  **Visualization Loop:**
        * **Code:**
            ```python
            for i, sample_idx in enumerate(grad_cam_indices):
                input_tensor = grad_cam_samples[i].unsqueeze(0)
                input_tensor.requires_grad = True # CRITICAL for gradient calculation
                model.zero_grad() # Clear any stale gradients

                # Forward pass to get output AND trigger hooks
                output, _ = model(input_tensor)
                predicted_class = torch.argmax(output, dim=1).item()

                # Calculate Grad-CAM (includes backward pass inside)
                heatmap = get_grad_cam(model, model.feature_maps, predicted_class, output)

                if heatmap is not None:
                    # Upsample heatmap to original input size
                    heatmap_tensor = torch.tensor(heatmap).unsqueeze(0).unsqueeze(0)
                    upsampled_heatmap = F.interpolate(heatmap_tensor, size=INPUT_SIZE, mode='linear', align_corners=False)
                    upsampled_heatmap = upsampled_heatmap.squeeze().numpy()

                    # Plotting
                    plt.figure(...)
                    original_data = input_tensor.squeeze().detach().cpu().numpy()
                    plt.plot(original_data, ...)
                    plt.imshow(upsampled_heatmap[np.newaxis, :], cmap='viridis', aspect='auto', alpha=0.5,
                               extent=[0, INPUT_SIZE -1, np.min(original_data)-0.1, np.max(original_data)+0.1])
                    # ... (colorbar, title, save, show) ...

                # Clean up
                model.gradients = None
                model.feature_maps = None
                input_tensor.requires_grad = False
            ```
        * **Explanation:**
            * Iterates through selected test samples.
            * Sets `requires_grad=True` on the input tensor because we need to compute gradients *from* the output *back towards* the intermediate CNN layer via the input.
            * `model.zero_grad()` ensures gradients from previous iterations don't interfere.
            * The `model(input_tensor)` call performs the forward pass, which gets the prediction *and* triggers the hooks to store `model.feature_maps`.
            * `get_grad_cam(...)` is called. Inside this function, the `backward()` call triggers the gradient calculation and the storage of `model.gradients` via the hook.
            * **Upsampling:** The calculated `heatmap` has a length corresponding to the sequence length *after* the CNN pooling layers (`cnn_output_seq_len`). To visualize it relative to the original input features (length `INPUT_SIZE`), `F.interpolate` is used to resize the heatmap linearly.
            * **Plotting:** The original feature sequence is plotted. The `upsampled_heatmap` is plotted *over* it using `plt.imshow`. `aspect='auto'` stretches the heatmap horizontally. `alpha=0.5` makes it semi-transparent. `extent` is crucial: `[0, INPUT_SIZE - 1, y_min, y_max]` tells `imshow` to map the heatmap columns to the range 0 to `INPUT_SIZE - 1` on the x-axis (aligning with the feature plot) and adjusts the y-limits for visual clarity.
            * **Cleanup:** Resetting hooks and `requires_grad` prevents potential issues in subsequent iterations or model uses.

---

### 3. Attention Weights Visualization

* **Purpose:** To visualize the internal mechanism of the Attention layer itself. In this model, it's a `MultiheadAttention` layer operating on the sequence output by the LSTM. Visualizing the weights shows which elements (time steps or positions) in the LSTM output sequence the attention mechanism focused on *relative to other elements* when computing the weighted representation (`attn_output`) that gets aggregated into the `context_vector`. For self-attention, it answers "How much did sequence element *i* attend to sequence element *j*?".
* **Concept:** Attention mechanisms compute scores indicating the relevance between elements. For self-attention (`query`, `key`, and `value` all come from the same sequence), the layer calculates how much each position ("query") should pay attention to every other position including itself ("key"). These scores are typically normalized (e.g., via Softmax) to become weights that sum to 1 for each query position. A higher weight means higher attention/focus. Visualizing these weights as a matrix (heatmap) reveals these focus patterns.
* **Step-by-Step Implementation (from the script):**

    1.  **Get Attention Weights:**
        * **Code:**
            ```python
            with torch.no_grad(): # No gradients needed for this
                output, attn_weights = model(input_tensor)
            # attn_weights shape likely [batch, seq_len_after_cnn, seq_len_after_cnn]
            # or [batch, num_heads, seq_len_after_cnn, seq_len_after_cnn]
            ```
        * **Explanation:** The model's `forward` pass is executed within `torch.no_grad()` as only the forward output (specifically `attn_weights`) is needed. The `attn_weights` tensor is directly returned by the `model`. *Note:* The exact shape depends on the `nn.MultiheadAttention` implementation details and how it's used. The script assumes the weights available for visualization are shaped `[batch, seq_len, seq_len]`. If `num_heads > 1`, the script implicitly either uses an attention layer that averages heads internally or it might be visualizing only one head or an average (though the averaging code is commented out in the model definition but might be intended). Let's proceed assuming `attn_weights` has the shape `[batch, seq_len, seq_len]`.

    2.  **Process Weights:**
        * **Code:**
            ```python
            if attn_weights is not None:
                # Squeeze batch dim, move to CPU, convert to NumPy
                attn_map = attn_weights.squeeze(0).cpu().numpy() # Shape: [seq_len, seq_len]
            ```
        * **Explanation:** Checks if weights were returned. `squeeze(0)` removes the batch dimension (assuming batch size 1 for individual explanation). `.cpu().numpy()` moves the tensor off the GPU (if used) and converts it to a NumPy array suitable for plotting libraries like Seaborn/Matplotlib. The resulting `attn_map` is a 2D array where `attn_map[i, j]` represents the attention paid by query step `i` to key step `j`.

    3.  **Visualize Heatmap:**
        * **Code:**
            ```python
            plt.figure(figsize=(7, 6))
            sns.heatmap(attn_map, cmap="viridis", cbar=True)
            plt.title(f'Attention Map for Sample {sample_idx} (Predicted: {CLASS_NAMES[predicted_class]})')
            plt.xlabel('Key Time Steps (CNN Output Sequence)')
            plt.ylabel('Query Time Steps (CNN Output Sequence)')
            plt.savefig(f"attention_map_sample_{sample_idx}.png")
            plt.show()
            ```
        * **Explanation:**
            * `seaborn.heatmap` is used to plot the 2D `attn_map`.
            * The y-axis represents the "query" positions (the position generating the attention context).
            * The x-axis represents the "key" positions (the positions being attended to).
            * Both axes correspond to the sequence length *after* the CNN layers (i.e., the input sequence to the attention layer).
            * Brighter colors indicate higher attention weights. A bright diagonal might indicate strong self-attention. Off-diagonal bright spots show attention paid to other positions in the sequence. Different patterns (e.g., focus on early/late steps, specific relative positions) can reveal how the model integrates information across the sequence.

---

In summary:
* **SHAP:** Explains prediction based on **input features**. Answers "Which input features mattered most?".
* **Grad-CAM:** Explains prediction based on **CNN spatial focus**. Answers "Where did the CNN look?".
* **Attention Viz:** Explains prediction based on **Attention layer's internal weighting**. Answers "Which parts of the sequence representation did the Attention layer focus on?".

  

<div align="center">
  <a href="https://maazsalman.org/">
    <img width="70" src="click-svgrepo-com.svg" alt="gh" />
  </a>
  <p> Explore More! 🚀</p>
</div>
