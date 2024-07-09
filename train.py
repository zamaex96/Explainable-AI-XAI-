import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
from data_loader import CustomDataset  # Adjust import as needed
import shap

# Model parameters
input_size = 4
cnn_channels = 16
lstm_hidden_size = 32
lstm_num_layers = 2
output_size = 4

# Define HybridCNNLSTM Model
class HybridCNNLSTM(nn.Module):
    def __init__(self, input_size, cnn_channels, lstm_hidden_size, lstm_num_layers, output_size):
        super(HybridCNNLSTM, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=cnn_channels, out_channels=cnn_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.lstm = nn.LSTM(input_size=cnn_channels * 2, hidden_size=lstm_hidden_size, num_layers=lstm_num_layers, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_size, output_size)

    def forward(self, x):
        x = x.unsqueeze(1)  # Reshape to [batch_size, 1, input_size]
        x = self.cnn(x)
        x = x.permute(0, 2, 1)  # Reshape to [batch_size, seq_len, cnn_channels * 2]
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])  # Use the last LSTM output for classification
        return x

# Instantiate the model, criterion, and optimizer
model = HybridCNNLSTM(input_size, cnn_channels, lstm_hidden_size, lstm_num_layers, output_size)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Load datasets using DataLoader
train_csv_path = "train_set.csv"
test_csv_path = "test_set.csv"

train_dataset = CustomDataset(train_csv_path)
test_dataset = CustomDataset(test_csv_path)

train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Training parameters
epochs = 200
train_loss_values = []
test_loss_values = []
train_accuracy_values = []
test_accuracy_values = []

# SHAP explainer
background_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
background_data = next(iter(background_loader))[0].to(device)  # Take a batch of training data for SHAP
explainer = shap.DeepExplainer(model, background_data)

# Training loop
for epoch in range(epochs):
    # Training phase
    model.train()
    epoch_train_loss = 0.0
    correct_train = 0
    total_train = 0
    for inputs, labels in train_data_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
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

    # Testing phase
    model.eval()
    epoch_test_loss = 0.0
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for inputs, labels in test_data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            epoch_test_loss += loss.item()

            _, predicted_test = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted_test == labels).sum().item()

    epoch_test_loss /= len(test_data_loader)
    test_loss_values.append(epoch_test_loss)
    test_accuracy = 100 * correct_test / total_test
    test_accuracy_values.append(test_accuracy)

    if (epoch + 1) % 5 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {epoch_train_loss:.4f}, Test Loss: {epoch_test_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%')

        # Explain a batch of test samples
        sample_inputs, _ = next(iter(test_data_loader))
        sample_inputs = sample_inputs.to(device)
        shap_values = explainer.shap_values(sample_inputs)
        shap.summary_plot(shap_values, sample_inputs.cpu().numpy(), feature_names=["Feature1", "Feature2", "Feature3", "Feature4"])

# Save the trained model
torch.save(model.state_dict(), 'saved_model_hybrid.pth')

# Save training information to CSV
train_info = {'train_loss': train_loss_values,
              'train_accuracy': train_accuracy_values,
              'test_loss': test_loss_values,
              'test_accuracy': test_accuracy_values}

train_info_df = pd.DataFrame(train_info)
train_info_df.to_csv("train_loss_hybrid.csv", index=False)

# Plot the loss and accuracy on the same figure
plt.figure(figsize=(12, 4))

# Loss plot
plt.subplot(2, 1, 1)
plt.plot(range(1, epochs + 1), train_loss_values, label='Training Loss')
plt.plot(range(1, epochs + 1), test_loss_values, label='Testing Loss')
plt.title('Training and Testing Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Accuracy plot
plt.subplot(2, 1, 2)
plt.plot(range(1, epochs + 1), train_accuracy_values, label='Training Accuracy')
plt.plot(range(1, epochs + 1), test_accuracy_values, label='Testing Accuracy')
plt.title('Training and Testing Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.savefig("train_accuracy_loss_hybrid_plot.eps", format='eps')
plt.show()
