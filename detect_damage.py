# detect_damage.py
import torch
from torchvision import transforms
from PIL import Image
import os
import json

# Cargar el modelo CNN entrenado
model = models.efficientnet_b0(pretrained=False)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)  # 5 clases
model.load_state_dict(torch.load("models/cnn_efficientnet.pth"))
model.eval()

# Transformaciones de datos
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Función para predecir
def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Añadir dimensión de lote
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        return predicted.item()

# Probar con un conjunto de datos de prueba
test_dir = "C:/VND_AI/data/test"  # Carpeta con imágenes de prueba
correct = 0
total = 0

for img_name in os.listdir(test_dir):
    if img_name.endswith(('jpg', 'png', 'jpeg')):
        img_path = os.path.join(test_dir, img_name)
        true_label = load_labels(img_path)  # Función para cargar etiquetas
        predicted_label = predict(img_path)
        
        if predicted_label == true_label:
            correct += 1
        total += 1

# Calcular precisión
accuracy = 100 * correct / total
print(f"Accuracy on test set: {accuracy:.2f}%")