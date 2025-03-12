import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import transforms
import os
from PIL import Image
import numpy as np
import cv2
import re

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
batch_size = 32
input_channels = 1
epochs = 50
learning_rate = 0.001
num_classes = 26
image_size = 224
patch_size = 16
num_heads = 8
transformer_layers = 6
mlp_dim = 512
hidden_dim = 256
train_split = 0.8

# Custom Dataset for the new directory structure
class SignLanguageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {chr(65+i): i for i in range(26)}  # A-Z mapping to 0-25
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        # Walk through user directories
        for user_dir in os.listdir(root_dir):
            user_path = os.path.join(root_dir, user_dir)
            if os.path.isdir(user_path):
                # Process all images in user directory
                for img_name in os.listdir(user_path):
                    if img_name.endswith(('.jpg', '.jpeg', '.png')):
                        # Extract letter from filename (assumes format like 'A0.jpg')
                        letter = img_name[0].upper()
                        if letter in self.class_to_idx:
                            img_path = os.path.join(user_path, img_name)
                            self.samples.append((img_path, self.class_to_idx[letter]))
        
        print(f"Loaded {len(self.samples)} images from {len(set(os.path.dirname(s[0]) for s in self.samples))} users")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# Data transforms
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Load dataset
print("Loading dataset...")
full_dataset = SignLanguageDataset(root_dir='./data/Dataset/', transform=transform)

# Split dataset
train_size = int(train_split * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

print(f"Training samples: {len(train_dataset)}")
print(f"Testing samples: {len(test_dataset)}")

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Vision Transformer Model (same as before)
class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x

class SignLanguageViT(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, num_classes, embed_dim, depth, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        n_patches = (img_size // patch_size) ** 2
        
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)
        x = x.transpose(0, 1)
        x = x[:, 0]
        x = self.mlp_head(x)
        return x

# Initialize model
model = SignLanguageViT(
    img_size=image_size,
    patch_size=patch_size,
    in_channels=input_channels,
    num_classes=num_classes,
    embed_dim=hidden_dim,
    depth=transformer_layers,
    num_heads=num_heads,
    mlp_dim=mlp_dim
)
model.to(device)

# Training functions (same as before)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# Training metrics
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

def train_epoch(model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if (batch_idx + 1) % 20 == 0:
            print(f'Batch: {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}')
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def evaluate(model, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(test_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc, all_preds, all_labels

# Live Camera Inference Class
class CameraInference:
    def __init__(self, model, transform, idx_to_class):
        self.model = model
        self.transform = transform
        self.idx_to_class = idx_to_class
        self.cap = None
        
    def preprocess_frame(self, frame):
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert to grayscale
        frame_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        # Convert to PIL Image
        pil_image = Image.fromarray(frame_gray)
        # Apply transforms
        tensor_image = self.transform(pil_image)
        # Add batch dimension
        tensor_image = tensor_image.unsqueeze(0)
        return tensor_image
        
    def start(self):
        self.model.eval()
        self.cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Preprocess frame
            tensor_frame = self.preprocess_frame(frame)
            tensor_frame = tensor_frame.to(device)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(tensor_frame)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                predicted_class = self.idx_to_class[predicted.item()]
                confidence = confidence.item()
            
            # Draw prediction on frame
            text = f'{predicted_class}: {confidence:.2f}'
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('Sign Language Recognition', frame)
            
            # Break loop with 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        self.stop()
        
    def stop(self):
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

# Training loop
print("Starting training...")
for epoch in range(epochs):
    print(f'\nEpoch {epoch+1}/{epochs}')
    
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    
    test_loss, test_acc, predictions, true_labels = evaluate(model, test_loader, criterion)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)
    
    scheduler.step()
    
    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')

# Generate final metrics and plots
final_predictions, final_labels = [], []
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        final_predictions.extend(predicted.cpu().numpy())
        final_labels.extend(labels.numpy())

# Print classification report
print("\nClassification Report:")
print(classification_report(final_labels, final_predictions, 
                          target_names=[chr(65+i) for i in range(26)]))

# Plot metrics
def plot_metrics():
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()

def plot_confusion_matrix():
    cm = confusion_matrix(final_labels, final_predictions)
    plt.figure(figsize=(15, 15))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[chr(65+i) for i in range(26)],
                yticklabels=[chr(65+i) for i in range(26)])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

# Generate plots
plot_metrics()
plot_confusion_matrix()

# Save model
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_losses': train_losses,
    'test_losses': test_losses,
    'train_accuracies': train_accuracies,
    'test_accuracies': test_accuracies,
}, 'sign_language_vit.pth')

# Start camera inference
print("\nStarting camera inference... Press 'q' to quit")
camera_inference = CameraInference(model, transform, full_dataset.idx_to_class)
camera_inference.start()
