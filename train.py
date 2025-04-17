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

# Assuming data_loader.py contains CustomDataset
# Make sure CustomDataset returns (data_tensor, label_tensor)
# Example:
# class CustomDataset(Dataset):
#     def __init__(self, csv_path, feature_cols=['Feature1', 'Feature2', 'Feature3', 'Feature4'], label_col='Label'):
#         self.data = pd.read_csv(csv_path)
#         self.features = self.data[feature_cols].values.astype(np.float32)
#         # Ensure labels are 0-indexed integers for CrossEntropyLoss
#         self.labels = self.data[label_col].astype(int).values 
#         # Optional: Store feature names
#         self.feature_names = feature_cols
#         # Optional: Store class names if available (replace with actual names)
#         self.class_names = [f"Class_{i}" for i in range(len(np.unique(self.labels)))] 

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

# --- Import your CustomDataset ---
from data_loader import CustomDataset # Adjust import as needed
# --- OR use the example definition above ---

# --- Configuration ---
INPUT_SIZE = 4          # Number of features
CNN_CHANNELS = 16
LSTM_HIDDEN_SIZE = 32
LSTM_NUM_LAYERS = 2
OUTPUT_SIZE = 4         # Number of classes
NUM_HEADS_ATTENTION = 4 # For MultiheadAttention
LEARNING_RATE = 0.005   # Adjusted learning rate
BATCH_SIZE = 32
EPOCHS = 50            # Reduced for quicker demo, increase for real training
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_CSV_PATH = "train_set.csv"
TEST_CSV_PATH = "test_set.csv"
SAVED_MODEL_PATH = 'saved_model_hybrid_explained.pth'
RESULTS_CSV_PATH = "train_results_hybrid_explained.csv"
PLOT_SAVE_PATH = "train_plots_hybrid_explained.eps"
NUM_SHAP_BACKGROUND_SAMPLES = 100 # Samples for SHAP background
NUM_SAMPLES_TO_EXPLAIN = 5      # How many test samples to explain in detail

# Define feature and class names (replace with actual names if known)
# Try to get from dataset if possible, otherwise define manually
try:
    temp_dataset = CustomDataset(TRAIN_CSV_PATH)
    FEATURE_NAMES = temp_dataset.feature_names
    CLASS_NAMES = temp_dataset.class_names
    del temp_dataset # free memory
except AttributeError:
    print("Warning: CustomDataset does not have feature_names/class_names attributes. Using default names.")
    FEATURE_NAMES = [f"Feature_{i+1}" for i in range(INPUT_SIZE)]
    CLASS_NAMES = [f"Class_{i}" for i in range(OUTPUT_SIZE)]


# --- Improved Model with Attention ---
class HybridCNNLSTMAttention(nn.Module):
    def __init__(self, input_size, cnn_channels, lstm_hidden_size, lstm_num_layers, output_size, num_heads):
        super(HybridCNNLSTMAttention, self).__init__()
        self.input_size = input_size

        # --- CNN Layers ---
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_channels), # Added Batch Norm
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=cnn_channels, out_channels=cnn_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_channels * 2), # Added Batch Norm
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # Calculate CNN output size dynamically (important if input_size changes)
        # We need the sequence length after CNN + MaxPool
        # Each MaxPool1d with kernel=2, stride=2 halves the sequence length
        # Note: For Conv1d, input shape is (N, C_in, L_in) -> output (N, C_out, L_out)
        # For LSTM, input shape is (N, L, H_in) (if batch_first=True)
        # Input to CNN: (Batch, 1, input_size)
        # After CNN block: (Batch, cnn_channels * 2, input_size / 4) -- Assuming input_size is divisible by 4
        self.cnn_output_seq_len = input_size // 4 # Calculate based on pooling layers
        lstm_input_features = cnn_channels * 2

        # --- LSTM Layers ---
        self.lstm = nn.LSTM(input_size=lstm_input_features,
                            hidden_size=lstm_hidden_size,
                            num_layers=lstm_num_layers,
                            batch_first=True,
                            bidirectional=True) # Using Bidirectional LSTM

        lstm_output_size = lstm_hidden_size * 2 # *2 because bidirectional

        # --- Attention Layer ---
        # Using MultiheadAttention
        self.attention = nn.MultiheadAttention(embed_dim=lstm_output_size,
                                                num_heads=num_heads,
                                                batch_first=True)

        # --- Final Classifier ---
        self.fc = nn.Linear(lstm_output_size, output_size) # Input is LSTM output size

        # --- For Grad-CAM ---
        self.feature_maps = None
        self.gradients = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        # Input x shape: [batch_size, input_size]
        batch_size = x.shape[0]
        
        # --- CNN Part ---
        x_cnn = x.unsqueeze(1)  # Reshape to [batch_size, 1, input_size] for Conv1d
        x_cnn = self.cnn(x_cnn) # Output: [batch_size, cnn_channels*2, seq_len_after_cnn]

        # Hook for Grad-CAM (target the output of the last conv layer's activation)
        if self.training is False and x_cnn.requires_grad: # Only hook during eval if grads are needed
             h = x_cnn.register_hook(self.activations_hook)
             self.feature_maps = x_cnn # Store feature maps

        # --- LSTM Part ---
        # Reshape for LSTM: [batch_size, seq_len, features]
        x_lstm = x_cnn.permute(0, 2, 1) # [batch_size, seq_len_after_cnn, cnn_channels*2]
        lstm_out, _ = self.lstm(x_lstm) # Output: [batch_size, seq_len_after_cnn, lstm_hidden_size * 2]

        # --- Attention Part ---
        # Use LSTM output for query, key, value in self-attention
        attn_output, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        # attn_output: [batch_size, seq_len_after_cnn, lstm_hidden_size * 2]
        # attn_weights: [batch_size, seq_len_after_cnn, seq_len_after_cnn]

        # We need a single vector per sequence. Aggregate attention outputs.
        # Option 1: Use the output corresponding to the last time step (like original LSTM)
        # context_vector = attn_output[:, -1, :]
        # Option 2: Average Pooling over sequence length
        context_vector = torch.mean(attn_output, dim=1)
        # Option 3: Max Pooling over sequence length
        # context_vector = torch.max(attn_output, dim=1)[0]

        # --- Classifier Part ---
        output = self.fc(context_vector) # Output: [batch_size, output_size]

        # Return attention weights for visualization if needed
        return output, attn_weights # Also return attention weights

# --- Instantiate Model, Loss, Optimizer ---
model = HybridCNNLSTMAttention(INPUT_SIZE, CNN_CHANNELS, LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, OUTPUT_SIZE, NUM_HEADS_ATTENTION)
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) # Using Adam optimizer
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=EPOCHS // 3, gamma=0.1) # Learning rate scheduler

# --- Load Data ---
train_dataset = CustomDataset(TRAIN_CSV_PATH)
test_dataset = CustomDataset(TEST_CSV_PATH)

train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False) # No shuffle for test

# --- Training Loop ---
train_loss_values = []
test_loss_values = []
train_accuracy_values = []
test_accuracy_values = []

print("Starting Training...")
for epoch in range(EPOCHS):
    # --- Training Phase ---
    model.train()
    epoch_train_loss = 0.0
    correct_train = 0
    total_train = 0
    for inputs, labels in train_data_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs, _ = model(inputs) # Ignore attention weights during training loss calculation
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item()
        _, predicted_train = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted_train == labels).sum().item()

    epoch_train_loss /= len(train_data_loader)
    train_loss_values.append(epoch_train_loss)
    train_accuracy = 100 * correct_train / total_train
    train_accuracy_values.append(train_accuracy)

    # --- Testing Phase ---
    model.eval()
    epoch_test_loss = 0.0
    correct_test = 0
    total_test = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_data_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            outputs, _ = model(inputs) # Ignore attention weights here too
            loss = criterion(outputs, labels)
            epoch_test_loss += loss.item()

            _, predicted_test = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted_test == labels).sum().item()
            
            all_preds.extend(predicted_test.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_test_loss /= len(test_data_loader)
    test_loss_values.append(epoch_test_loss)
    test_accuracy = 100 * correct_test / total_test
    test_accuracy_values.append(test_accuracy)
    
    scheduler.step() # Update learning rate

    if (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1: # Print every 10 epochs or last epoch
         print(f'Epoch [{epoch + 1}/{EPOCHS}], LR: {scheduler.get_last_lr()[0]:.5f}, Train Loss: {epoch_train_loss:.4f}, Test Loss: {epoch_test_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}%')

print("Training Finished.")

# --- Save Model and Results ---
torch.save(model.state_dict(), SAVED_MODEL_PATH)
print(f"Model saved to {SAVED_MODEL_PATH}")

train_info = {'epoch': range(1, EPOCHS + 1),
              'train_loss': train_loss_values,
              'train_accuracy': train_accuracy_values,
              'test_loss': test_loss_values,
              'test_accuracy': test_accuracy_values}
train_info_df = pd.DataFrame(train_info)
train_info_df.to_csv(RESULTS_CSV_PATH, index=False)
print(f"Training results saved to {RESULTS_CSV_PATH}")


# --- Plotting Loss and Accuracy ---
plt.figure(figsize=(12, 5))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(train_info_df['epoch'], train_info_df['train_loss'], label='Training Loss')
plt.plot(train_info_df['epoch'], train_info_df['test_loss'], label='Testing Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(train_info_df['epoch'], train_info_df['train_accuracy'], label='Training Accuracy')
plt.plot(train_info_df['epoch'], train_info_df['test_accuracy'], label='Testing Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(PLOT_SAVE_PATH, format='eps')
print(f"Training plots saved to {PLOT_SAVE_PATH}")
plt.show()


# --- Explainability Analysis ---
print("\n--- Starting Explainability Analysis ---")
model.eval() # Ensure model is in eval mode

# 1. Confusion Matrix
print("\n--- Confusion Matrix ---")
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig("confusion_matrix_hybrid.png")
plt.show()

print("\n--- Classification Report ---")
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))


# 2. SHAP Analysis
print("\n--- SHAP Analysis ---")
# Use a subset of training data or test data as background
# Using a subset of test data for background can sometimes be informative too
background_indices = random.sample(range(len(train_dataset)), min(NUM_SHAP_BACKGROUND_SAMPLES, len(train_dataset)))
background_data = torch.stack([train_dataset[i][0] for i in background_indices]).to(DEVICE)

# Create a wrapper function for SHAP if the model returns multiple outputs (like attention weights)
def model_predictor(data_tensor):
    output, _ = model(data_tensor)
    return output

explainer = shap.DeepExplainer(model_predictor, background_data)

# Explain a subset of test samples
test_indices = random.sample(range(len(test_dataset)), min(NUM_SAMPLES_TO_EXPLAIN * 5, len(test_dataset))) # Get more samples for summary
test_samples = torch.stack([test_dataset[i][0] for i in test_indices]).to(DEVICE)
test_labels = torch.tensor([test_dataset[i][1] for i in test_indices]).to(DEVICE)

print(f"Calculating SHAP values for {test_samples.shape[0]} test samples...")
# Need to detach from graph and potentially move to CPU if SHAP requires numpy
shap_values = explainer.shap_values(test_samples)
print("SHAP values calculated.")

# SHAP Summary Plot (Global Feature Importance) - across all classes
shap.summary_plot(shap_values, test_samples.cpu().numpy(), feature_names=FEATURE_NAMES,
                  class_names=CLASS_NAMES, plot_type="bar", max_display=10)
plt.title("SHAP Global Feature Importance (All Classes)")
plt.savefig("shap_summary_bar_all_classes.png")
plt.show()

# SHAP Summary Plot per Class (Dot plot)
for i, class_name in enumerate(CLASS_NAMES):
    plt.figure() # Create a new figure for each plot
    shap.summary_plot(shap_values[i], test_samples.cpu().numpy(), feature_names=FEATURE_NAMES,
                      show=False, max_display=10)
    plt.title(f"SHAP Feature Importance for {class_name}")
    plt.savefig(f"shap_summary_dot_{class_name}.png")
    plt.close() # Close the plot to avoid display issues in loops


# SHAP Force Plots for individual predictions
print(f"\n--- SHAP Force Plots for {NUM_SAMPLES_TO_EXPLAIN} individual samples ---")
# Detach expected value if it's a tensor
try:
    # Newer SHAP versions might require this adjustment for multi-output explainers
    base_values = explainer.expected_value.cpu().numpy() if isinstance(explainer.expected_value, torch.Tensor) else explainer.expected_value
    if isinstance(base_values, list): # Handle list case if DeepExplainer returns list
        base_values = np.array(base_values)
except AttributeError: # Older versions might not have expected_value directly accessible this way after shap_values calculation
     print("Warning: Could not automatically get SHAP base values. Force plots might be limited.")
     base_values = None # Set to None if unavailable

# Use the calculated shap_values (list of arrays) and test_samples (numpy array)
test_samples_np = test_samples.cpu().numpy()

# Select a few samples to explain
indices_to_plot = random.sample(range(len(test_indices)), NUM_SAMPLES_TO_EXPLAIN)

for i in indices_to_plot:
    sample_idx_in_original_test = test_indices[i]
    original_label = test_dataset[sample_idx_in_original_test][1].item()
    
    with torch.no_grad():
        output_prob = torch.softmax(model_predictor(test_samples[i].unsqueeze(0)), dim=1).squeeze()
    predicted_label = torch.argmax(output_prob).item()
    
    print(f"\nExplaining Test Sample #{sample_idx_in_original_test}:")
    print(f"  Features: {np.round(test_samples_np[i], 2)}")
    print(f"  True Class: {CLASS_NAMES[original_label]} ({original_label})")
    print(f"  Predicted Class: {CLASS_NAMES[predicted_label]} ({predicted_label}) - Prob: {output_prob[predicted_label]:.3f}")
    print(f"  Correct: {'Yes' if original_label == predicted_label else 'No'}")
    
    # Force plot for the predicted class
    if base_values is not None and len(base_values) == OUTPUT_SIZE:
        shap.force_plot(base_values[predicted_label],
                        shap_values[predicted_label][i,:],
                        test_samples_np[i,:],
                        feature_names=FEATURE_NAMES,
                        matplotlib=True, show=False) # Use matplotlib=True for saving
        plt.title(f'SHAP Force Plot for Sample {sample_idx_in_original_test} (Predicted: {CLASS_NAMES[predicted_label]})')
        plt.savefig(f"shap_force_plot_sample_{sample_idx_in_original_test}_pred_{predicted_label}.png")
        plt.close()
        
        # Optional: Force plot for the true class if different from predicted
        if original_label != predicted_label:
             shap.force_plot(base_values[original_label],
                        shap_values[original_label][i,:],
                        test_samples_np[i,:],
                        feature_names=FEATURE_NAMES,
                        matplotlib=True, show=False)
             plt.title(f'SHAP Force Plot for Sample {sample_idx_in_original_test} (True Class: {CLASS_NAMES[original_label]})')
             plt.savefig(f"shap_force_plot_sample_{sample_idx_in_original_test}_true_{original_label}.png")
             plt.close()

    else:
         print("  Skipping force plot generation (base values unavailable or mismatch).")


# 3. Grad-CAM for CNN Layer visualization
print("\n--- Grad-CAM Analysis ---")

def get_grad_cam(model, target_layer_output, target_class_index, output):
    """Calculates Grad-CAM heatmap"""
    if model.gradients is None or target_layer_output is None:
        print("Warning: Gradients or feature maps not captured. Cannot compute Grad-CAM.")
        return None
        
    # Get gradients w.r.t the target class score
    output[:, target_class_index].backward(retain_graph=True) # Backpropagate the specific class score
    
    gradients = model.gradients # Shape: [batch, channels, seq_len]
    activations = target_layer_output # Shape: [batch, channels, seq_len]

    # Pool gradients across the sequence length (alpha_k calculation)
    pooled_gradients = torch.mean(gradients, dim=[2]) # Shape: [batch, channels]

    # Weight the channels by corresponding gradients (importance weighting)
    # Add dimensions for broadcasting: [batch, channels, 1]
    pooled_gradients = pooled_gradients.unsqueeze(-1)
    
    # Weighted sum of activations: [batch, channels, seq_len] * [batch, channels, 1] -> sum over channels
    heatmap = torch.sum(activations * pooled_gradients, dim=1) # Shape: [batch, seq_len]
    heatmap = F.relu(heatmap) # Apply ReLU

    # Normalize the heatmap
    if heatmap.ndim == 1: # Handle case for batch size 1
        heatmap = heatmap.unsqueeze(0)
    
    max_val = torch.max(heatmap, dim=1, keepdim=True)[0]
    min_val = torch.min(heatmap, dim=1, keepdim=True)[0]
    # Add epsilon to avoid division by zero
    heatmap = (heatmap - min_val) / (max_val - min_val + 1e-8)

    return heatmap.squeeze().cpu().numpy() # Return as numpy array

# Select a few samples for Grad-CAM
grad_cam_indices = random.sample(range(len(test_dataset)), NUM_SAMPLES_TO_EXPLAIN)
grad_cam_samples = torch.stack([test_dataset[i][0] for i in grad_cam_indices]).to(DEVICE)
grad_cam_labels = torch.tensor([test_dataset[i][1] for i in grad_cam_indices])

for i, sample_idx in enumerate(grad_cam_indices):
    input_tensor = grad_cam_samples[i].unsqueeze(0) # Add batch dimension
    input_tensor.requires_grad = True # Need gradients for input
    model.zero_grad() # Reset gradients

    # Forward pass to get output and capture activations/gradients
    output, _ = model(input_tensor)
    predicted_class = torch.argmax(output, dim=1).item()

    # Calculate Grad-CAM for the predicted class
    # Need to run forward pass again within the Grad-CAM function's scope
    # OR ensure hooks captured correctly. Let's re-run here to be safe.
    model.zero_grad()
    input_tensor.requires_grad = True
    output, _ = model(input_tensor) # This pass will trigger the hooks

    heatmap = get_grad_cam(model, model.feature_maps, predicted_class, output)
    
    if heatmap is not None:
        # --- Visualization ---
        # The heatmap corresponds to the *output* sequence of the CNN layers.
        # We need to visualize it relative to the *input* sequence.
        # Option 1: Plot heatmap directly (length = input_size / 4)
        # Option 2: Upsample heatmap to original input_size
        
        # Upsample heatmap to original input size
        heatmap_tensor = torch.tensor(heatmap).unsqueeze(0).unsqueeze(0) # [1, 1, seq_len]
        upsampled_heatmap = F.interpolate(heatmap_tensor, size=INPUT_SIZE, mode='linear', align_corners=False)
        upsampled_heatmap = upsampled_heatmap.squeeze().numpy()

        plt.figure(figsize=(12, 4))
        original_data = input_tensor.squeeze().detach().cpu().numpy()
        
        # Plot original features
        plt.plot(original_data, label=f'Features (Sample {sample_idx})', color='blue', alpha=0.6)
        
        # Overlay Grad-CAM heatmap
        plt.imshow(upsampled_heatmap[np.newaxis, :], cmap='viridis', aspect='auto', alpha=0.5,
                   extent=[0, INPUT_SIZE -1, np.min(original_data)-0.1, np.max(original_data)+0.1]) # Adjust y-limits based on data range
                   
        plt.colorbar(label='Grad-CAM Intensity')
        plt.title(f'Grad-CAM for Sample {sample_idx} (Predicted: {CLASS_NAMES[predicted_class]}, True: {CLASS_NAMES[grad_cam_labels[i].item()]})')
        plt.xlabel('Feature Dimension / Time Step')
        plt.ylabel('Feature Value')
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig(f"grad_cam_sample_{sample_idx}.png")
        plt.show()
        
    # Clean up gradients and hooks potentially? (Good practice)
    model.gradients = None
    model.feature_maps = None
    input_tensor.requires_grad = False


# 4. Attention Weights Visualization (Optional but recommended)
print("\n--- Attention Weights Analysis ---")
# Select a few samples again
attn_indices = random.sample(range(len(test_dataset)), NUM_SAMPLES_TO_EXPLAIN)
attn_samples = torch.stack([test_dataset[i][0] for i in attn_indices]).to(DEVICE)
attn_labels = torch.tensor([test_dataset[i][1] for i in attn_indices])

for i, sample_idx in enumerate(attn_indices):
    input_tensor = attn_samples[i].unsqueeze(0) # Add batch dimension
    
    with torch.no_grad():
      output, attn_weights = model(input_tensor)
    
    predicted_class = torch.argmax(output, dim=1).item()
    
    # attn_weights shape: [batch_size, num_heads, seq_len, seq_len] if using MultiheadAttention directly
    # OR [batch_size, seq_len, seq_len] if aggregated or custom attention
    # For MultiheadAttention, average over heads? Or pick one head? Let's average.
    # Check the shape returned by *your* specific attention implementation.
    # Assuming model returns [batch, seq_len_after_cnn, seq_len_after_cnn] from MultiheadAttention
    
    if attn_weights is not None:
        # Squeeze batch dim, avg over heads if necessary (check shape)
        # Example for MultiheadAttention: attn_weights = attn_weights.squeeze(0).mean(dim=0) # Avg heads -> [seq_len, seq_len]
        # If output is directly [batch, seq_len, seq_len]:
        attn_map = attn_weights.squeeze(0).cpu().numpy() # Shape: [seq_len, seq_len]

        plt.figure(figsize=(7, 6))
        sns.heatmap(attn_map, cmap="viridis", cbar=True)
        plt.title(f'Attention Map for Sample {sample_idx} (Predicted: {CLASS_NAMES[predicted_class]})')
        plt.xlabel('Key Time Steps (CNN Output Sequence)')
        plt.ylabel('Query Time Steps (CNN Output Sequence)')
        plt.savefig(f"attention_map_sample_{sample_idx}.png")
        plt.show()

        # Sometimes visualizing the attention paid *by the last time step* (or the aggregated context vector)
        # to all previous steps is useful. This depends on how attention is used before the FC layer.
        # If context vector is mean/max pooled (as in this example), visualizing the attention weights 
        # associated with *that* aggregated vector isn't straightforward from attn_weights directly.
        # The heatmap above shows step-to-step attention within the sequence.


print("\nExplainability Analysis Complete.")
