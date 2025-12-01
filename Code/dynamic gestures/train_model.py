import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse

# --- Custom Dataset for Sequences ---
class GestureSequenceDataset(Dataset):
    def __init__(self, data_dir):
        self.class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        self.data = []
        self.labels = []

        for i, class_name in enumerate(self.class_names):
            class_path = os.path.join(data_dir, class_name)
            for file_name in os.listdir(class_path):
                if file_name.endswith('.npy'):
                    sequence = np.load(os.path.join(class_path, file_name))
                    sequence -= sequence[0] # Normalize
                    self.data.append(torch.tensor(sequence, dtype=torch.float32))
                    self.labels.append(i)
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx], self.labels[idx]

# --- LSTM Model ---
class GestureLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(GestureLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n.squeeze(0))

# --- Main Training Logic ---
def train_dynamic_model(args):
    data_dir = os.path.join("dynamic_dataset", args.hand)
    model_save_path = "custom_models"
    os.makedirs(model_save_path, exist_ok=True)

    dataset = GestureSequenceDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    class_names = dataset.class_names
    num_classes = len(class_names)
    print(f"--- Training model for {args.hand.upper()} HAND ---")
    print(f"Found {num_classes} gestures: {class_names}")

    # The input size depends on whether we are tracking one hand (2) or two (4)
    input_size = 2 # Since we are training one model per hand
    model = GestureLSTM(input_size=input_size, hidden_size=64, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 50
    for epoch in range(num_epochs):
        for sequences, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

    print("Finished Training!")
    
    model_filename = f"dynamic_gesture_model_{args.hand}_hand.pth"
    class_filename = f"dynamic_class_names_{args.hand}_hand.txt"

    torch.save(model.state_dict(), os.path.join(model_save_path, model_filename))
    print(f"Model saved to {os.path.join(model_save_path, model_filename)}")

    with open(os.path.join(model_save_path, class_filename), "w") as f:
        f.write("\n".join(class_names))
    print(f"Class names saved to {os.path.join(model_save_path, class_filename)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a dynamic gesture model for a specific hand.")
    parser.add_argument("--hand", type=str, required=True, choices=['left', 'right'], help="The hand to train a model for.")
    args = parser.parse_args()
    train_dynamic_model(args)