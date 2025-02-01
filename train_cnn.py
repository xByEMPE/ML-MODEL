# train_cnn.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from PIL import Image
import os
import json

# Configuración
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = "C:/VND_AI/data/labels"
batch_size = 32
epochs = 40
learning_rate = 0.001

# Mapeo de etiquetas a números
label_mapping = {
    "Erosion - Laminate damage": 0,
    "Crack": 1,
    "Delamination": 2,
    "Impact damage": 3,
    "Trailing edge - Scratch - Scratch/Gouge": 4,
    "Bonding line - Debonding": 5,
    "Blade collar- Misalignment": 6,
    "Laminate damage": 7,
    "Other": 8
}

# Transformaciones de datos con redimensionamiento y padding
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Redimensionar a 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Cargar etiquetas desde archivos JSON
def load_labels(image_path):
    json_path = os.path.splitext(image_path)[0] + ".json"
    with open(json_path, "r") as f:
        data = json.load(f)
        
        # Extraer todas las etiquetas de "shapes"
        labels = [shape["label"] for shape in data["shapes"]]
        
        # Convertir las etiquetas a números usando el mapeo
        numeric_labels = [label_mapping[label] for label in labels]
        
        # Si hay múltiples etiquetas, elegimos la primera como la principal
        return numeric_labels[0] if numeric_labels else -1  # -1 indica "sin etiqueta"

# Dataset etiquetado
class LabeledDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(('jpg', 'png', 'jpeg'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = load_labels(img_path)
        if self.transform:
            image = self.transform(image)
        return image, label

# Filtrar imágenes sin etiquetas válidas
dataset = LabeledDataset(root_dir=data_dir, transform=transform)
filtered_dataset = [data for data in dataset if data[1] != -1]  # Excluir imágenes sin etiquetas

# DataLoader
train_loader = DataLoader(filtered_dataset, batch_size=batch_size, shuffle=True)

# Modelo CNN (usando EfficientNet)
model = models.efficientnet_b0(pretrained=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)  # 5 clases de daño
model = model.to(device)

# Función de pérdida y optimizador
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Entrenamiento
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Calcular precisión
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    # Imprimir métricas
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

# Guardar el modelo CNN
torch.save(model.state_dict(), "models/cnn_efficientnet.pth")