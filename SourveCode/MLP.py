import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
from multiprocessing import freeze_support

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 1024)
        self.bn1 = nn.BatchNorm1d(1024)     
        self.fc2 = nn.Linear(1024, 512)     
        self.bn2 = nn.BatchNorm1d(512)              
        self.fc3 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.5)      
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)   
        # Layer 1 
        x = self.fc1(x)
        x = self.bn1(x) 
        x = self.relu(x)
        x = self.dropout(x)
        # Layer 2 
        x = self.fc2(x)
        x = self.bn2(x) 
        x = self.relu(x)
        x = self.dropout(x)
        # Output Layer (fc3)
        x = self.fc3(x)
        return x
    
# Predict and display images
def predict_images(images, model, num_images=5):
    fig = plt.figure(figsize=(15, 3))
    for i in range(num_images):
        img, label = images[i]
        img_tensor = img.unsqueeze(0).to(device)
        with torch.no_grad():
            model.eval() 
            out = model(img_tensor)
            _, preds = torch.max(out, dim=1)

        # Show images
        ax = fig.add_subplot(1, num_images, i+1)
        img_show = img.permute(1, 2, 0).numpy()
        img_show = img_show * 0.5 + 0.5 
        ax.imshow(img_show)
        ax.axis('off')

        # Show label and prediction
        pred_class = classes[preds[0].item()]
        true_class = classes[label]
        color = 'green' if pred_class == true_class else 'red'
        ax.set_title(f'True: {true_class}\nPred: {pred_class}', color=color)

    plt.tight_layout()
    plt.show()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transformations 
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(), 
    transforms.RandomCrop(32, padding=4, padding_mode='reflect'), 
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), 
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train) 
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
# Split train into train and validation
val_size = int(0.1 * len(train_set))
train_size = len(train_set) - val_size
train_set, val_set = torch.utils.data.random_split(train_set, [train_size, val_size])

batch_size = 256 
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def evaluate_model(model, loader, criterion):
    model.eval() 
    loss_cnt = 0.0
    accuracy = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            loss = criterion(output, labels)

            loss_cnt += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()

    loss = loss_cnt / len(loader)
    acc = 100 * accuracy / total
    return loss, acc

# function to plot Confusion Matrix 
def plot_confusion_matrix(all_labels, all_preds, classes, model_name="Model"):
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.savefig(f'{model_name}_Confusion_matrix.png')
    plt.show()

# Train model
def train_model(model, train_loader, val_loader, num_epochs=50, model_name='Model'): 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    # OneCycleLR scheduler
    scheduler = OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_loader), epochs=num_epochs)

    train_counter = []
    train_losses = []
    train_accs = []

    val_counter = []
    val_losses= []
    val_accs = []

    # Early Stopping parameters
    best_val_acc = -1.0
    epochs_no_improve = 0
    limit_epoch = 10 # The limited epoches in which the epoches have no improvement compared with the previous one 

    for epoch in range(num_epochs):
        model.train()
        loss_cnt = 0.0
        accuracy= 0
        total= 0

        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [{model_name}]'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss_val = criterion(outputs, labels)
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            scheduler.step() 
            loss_cnt+= loss_val.item()
            _, predicted = torch.max(outputs.data, 1)
            total+= labels.size(0)
            accuracy += (predicted == labels).sum().item()

        epoch_train_loss = loss_cnt / len(train_loader)
        epoch_train_acc = 100 * accuracy / total
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)
        train_counter.append((epoch + 1) * len(train_loader.dataset))

        val_loss, val_acc = evaluate_model(model, val_loader, criterion)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_counter.append((epoch + 1) * len(train_loader.dataset)) 

        print(f'Epoch {epoch+1}: Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        # Early Stopping Check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == limit_epoch:
                print(f"Stop after {limit_epoch} epochs with no improvement.")
                break

    print(f"Finished Training")

    # --- Plotting Learning Curves (Loss) ---
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(train_counter, train_losses, color='blue', label='Training Loss')
    plt.plot(val_counter, val_losses, color='red', label='Validation Loss')
    plt.legend(loc='upper right')
    plt.xlabel('Number of training examples seen')
    plt.ylabel('Loss (Cross-Entropy Loss)')
    plt.title(f'Learning Curve: {model_name} (Train vs Validation Loss)')
    plt.grid(True)
    # --- Plotting Learning Curves (Accuracy) ---
    plt.subplot(1, 2, 2)
    plt.plot(train_counter, train_accs, color='blue', label='Training Accuracy')
    plt.plot(val_counter, val_accs, color='red', label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.xlabel('Number of training examples seen')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Learning Curve: {model_name} (Train vs Validation Accuracy)')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'{model_name}_Learning_curves_loss_acc.png')
    plt.show()

    return model

if __name__ == '__main__':
    freeze_support() 
    mlp_model = MLP().to(device)

    num_epochs = 50

    mlp_model = train_model(mlp_model, train_loader, val_loader, num_epochs=num_epochs, model_name='MLP')

    criterion = nn.CrossEntropyLoss()

    # Evaluate MLP on test set
    print("\n Evaluation for MLP on test set...")
    mlp_preds= []
    mlp_labels = []
    mlp_model.eval()
    loss_cnt = 0.0
    accuracy = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = mlp_model(images)
            loss = criterion(outputs, labels)
            loss_cnt += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            accuracy += (predicted == labels).sum().item()
            total += labels.size(0)
            mlp_preds.extend(predicted.cpu().numpy())
            mlp_labels.extend(labels.cpu().numpy())
    test_loss = loss_cnt / len(test_loader)
    test_acc = 100 * accuracy/total
    print(f"MLP Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
    plot_confusion_matrix(mlp_labels,mlp_preds, classes, model_name='MLP_Test')

    # Get some images to test
    test_images_for_pred = [test_set[i] for i in range(5)]
    predict_images(test_images_for_pred, mlp_model)
