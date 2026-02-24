# Develop a Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images.

##   PROBLEM STATEMENT AND DATASET
Include the Problem Statement and Dataset.

## Neural Network Model
| Layer   | Number of Nodes  |
|---------|------------------|
| Input   | 784              |
| Conv1   | 32 filters       |
| Conv2   | 64 filters       |
| Conv3   | 128 filters      |
| FC1     | 128              |
| FC2     | 64               |
| Output  | 10               |
<img width="950" height="534" alt="image" src="https://github.com/user-attachments/assets/a59e4cee-29aa-4fd9-a7c8-f87a134ec202" />



## DESIGN STEPS
## Step 1: Data Loading and Preprocessing
1. Import required libraries.
2. Define image transformations (ToTensor and Normalize).
3. Load the Fashion-MNIST training and test datasets.
4. Create DataLoader objects for batch processing.

## Step 2: Model Construction
1. Define the CNN architecture.
2. Add convolution layers with ReLU activation.
3. Apply max pooling layers.
4. Flatten the feature maps.
5. Add fully connected layers.
6. Define loss function and optimizer.

## Step 3: Model Training
1. Set the model to training mode.
2. Perform forward propagation.
3. Compute loss.
4. Perform backpropagation.
5. Update model weights.
6. Repeat for required epochs.

## Step 4: Model Evaluation and Prediction
1. Set the model to evaluation mode.
2. Calculate test accuracy.
3. Generate confusion matrix and classification report.
4. Predict the class of a single test image.

## PROGRAM

### Name: Surya Prakash B

### Register Number: 212224230281

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

## Step 1: Load and Preprocess Data
# Define transformations for images
transform = transforms.Compose([
    transforms.ToTensor(),          # Convert images to tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize images
])
# Load Fashion-MNIST dataset
train_dataset = torchvision.datasets.FashionMNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.FashionMNIST(root="./data", train=False, transform=transform, download=True)
# Get the shape of the first image in the training dataset
image, label = train_dataset[0]
print(image.shape)
print(len(train_dataset))
# Get the shape of the first image in the test dataset
image, label = test_dataset[0]
print(image.shape)
print(len(test_dataset))
# Create DataLoader for batch processing
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1)
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.fc1 = nn.Linear(128*3*3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
from torchsummary import summary
import torch
import torch.nn as nn

class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1)
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.fc1 = nn.Linear(128*3*3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize model
model = CNNClassifier()

# Move model to GPU if available
if torch.cuda.is_available():
    device = torch.device("cuda")
    model.to(device)

# Print model summary
print('Name: Surya Prakash B')
print('Register Number: 212224230281')
summary(model, input_size=(1, 28, 28))
# Initialize model, loss function, and optimizer
model = CNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)
## Step 3: Train the Model
def train_model(model, train_loader, num_epochs=3):
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for images, labels in train_loader:
              optimizer.zero_grad()
              outputs = model(images)
              loss = criterion(outputs, labels)
              loss.backward()
              optimizer.step()
              running_loss += loss.item()

        print('Name: Surya Prakash B')
        print('Register Number: 212224230281')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
# Train the model
train_model(model, train_loader)
## Step 4: Test the Model
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    print('Name: Surya Prakash B')
    print('Register Number: 212224230281')
    print(f'Test Accuracy: {accuracy:.4f}')

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    print('Name: Surya Prakash B')
    print('Register Number: 212224230281')
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_dataset.classes, yticklabels=test_dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Print classification report
    print('Name: Surya Prakash B')
    print('Register Number: 212224230281')
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))
# Evaluate the model
test_model(model, test_loader)
## Step 5: Predict on a Single Image
import matplotlib.pyplot as plt
def predict_image(model, image_index, dataset):
    model.eval()
    image, label = dataset[image_index]
    with torch.no_grad():
        output = model(image.unsqueeze(0))  # Add batch dimension
        _, predicted = torch.max(output, 1)
    class_names = dataset.classes

    # Display the image
    print('Name: Surya Prakash B')
    print('Register Number: 212224230281')
    plt.imshow(image.squeeze(), cmap="gray")
    plt.title(f'Actual: {class_names[label]}\nPredicted: {class_names[predicted.item()]}')
    plt.axis("off")
    plt.show()
    print(f'Actual: {class_names[label]}, Predicted: {class_names[predicted.item()]}')
# Example Prediction
predict_image(model, image_index=47, dataset=test_dataset)
```

## OUTPUT

## Training Loss per Epoch
<img width="1658" height="60" alt="image" src="https://github.com/user-attachments/assets/161c2ba8-b85e-4387-9a7d-91539dcb2908" />

## Confusion Matrix
<img width="712" height="89" alt="image" src="https://github.com/user-attachments/assets/c3e325ae-7da3-4b46-846e-e10e94d7f458" />
<img width="709" height="608" alt="image" src="https://github.com/user-attachments/assets/b639f1bc-65c9-4c19-90ea-f4bfe14fac80" />

## Classification Report
<img width="1664" height="296" alt="image" src="https://github.com/user-attachments/assets/e4357010-1bf4-4ffc-9789-e9f01664563b" />

### New Sample Data Prediction
<img width="405" height="491" alt="image" src="https://github.com/user-attachments/assets/cd00bca3-87d6-46fa-b67d-89dad2f38889" />


## RESULT
The Convolutional Neural Network (CNN) was successfully trained on the Fashion-MNIST dataset and achieved good classification accuracy on the test data.
The model correctly predicted most classes, as verified using accuracy score, confusion matrix, and classification report.
