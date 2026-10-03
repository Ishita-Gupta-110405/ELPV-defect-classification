import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from preprocessing import get_splits

class ELPVDataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.tensor(images, dtype=torch.float32).unsqueeze(1) / 255.0
        self.labels = torch.tensor(labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

def main():
    X_train, X_val, _, y_train, y_val, _ = get_splits()
    
    train_loader = DataLoader(ELPVDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(ELPVDataset(X_val, y_val), batch_size=32, shuffle=False)
    
    print("Setting up ResNet-18...")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    original_conv1 = model.conv1
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    with torch.no_grad():
        model.conv1.weight.copy_(original_conv1.weight.sum(dim=1, keepdim=True))
        
    model.fc = nn.Linear(model.fc.in_features, 4)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Running 1 preliminary epoch on {device}...")
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(inputs.to(device)), labels.to(device))
        loss.backward()
        optimizer.step()
        
    print("Evaluating ResNet-18...")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()
            
    print(f"\n--- ResNet-18 Preliminary Validation Accuracy: {100 * correct / total:.2f}% ---")

if __name__ == "__main__":
    main()