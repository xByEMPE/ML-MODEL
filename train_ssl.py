import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from PIL import Image
import os

# Configuración
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = "C:/VND_AI/data/no_labels"
batch_size = 32
epochs = 50
learning_rate = 0.001

# Transformaciones de datos (aumentación para SSL)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Dataset no etiquetado con dos vistas
class UnlabeledDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(('jpg', 'png', 'jpeg'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image1 = self.transform(image)
            image2 = self.transform(image)
        
        return image1, image2

# Cargar dataset
dataset = UnlabeledDataset(root_dir=data_dir, transform=transform)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Modelo SSL con EfficientNet
class SSLModel(nn.Module):
    def __init__(self):
        super(SSLModel, self).__init__()
        self.backbone = models.efficientnet_b0(pretrained=True)
        self.backbone.classifier = nn.Identity()  # Quitar capa final
        self.projector = nn.Sequential(
            nn.Linear(1280, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128)
        )

    def forward(self, x):
        features = self.backbone(x)
        projections = self.projector(features)
        return projections

model = SSLModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Descongelar la red para fine-tuning
for param in model.backbone.parameters():
    param.requires_grad = True

# Entrenamiento SSL
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    projection_variances = []
    
    for images1, images2 in train_loader:
        images1, images2 = images1.to(device), images2.to(device)
        
        optimizer.zero_grad()
        projections1 = model(images1)
        projections2 = model(images2)
        
        # Pérdida de contraste usando similitud de coseno
        loss = -torch.nn.functional.cosine_similarity(projections1, projections2, dim=1).mean()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Calcular varianza de las proyecciones
        batch_variance = projections1.var(dim=0).mean().item()
        projection_variances.append(batch_variance)
    
    # Imprimir métricas
    epoch_loss = running_loss / len(train_loader)
    epoch_variance = sum(projection_variances) / len(projection_variances)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Projection Variance: {epoch_variance:.6f}")

# Guardar el modelo SSL
torch.save(model.state_dict(), "models/ssl_efficientnet.pth")
