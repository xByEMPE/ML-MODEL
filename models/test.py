import os
import json
import numpy as np
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
import torchvision.transforms as T
from PIL import Image
import cv2

# --- CONFIGURACIÓN DEL DATASET Y ETIQUETAS ---
label_map = {
    "Categoria 1": 1,
    "Categoria 2": 2,
    "Categoria 3": 3,
    "Categoria 4": 4,
    "Categoria 5": 5
}
# Mapeo inverso para mostrar el nombre de la categoría en inferencia
label_names = {v: k for k, v in label_map.items()}

num_classes = len(label_map) + 1  # +1 para fondo

# --- 1. Definición del Dataset Personalizado (para entrenamiento) ---
class WindTurbineDamageDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        """
        Parámetros:
          - root: ruta raíz del dataset. Se espera la siguiente estructura:
                root/
                  images/        -> imágenes
                  annotations/   -> archivos JSON (con el mismo nombre base que la imagen)
          - transforms: transformaciones a aplicar a la imagen.
        """
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
    
    def __getitem__(self, idx):
        try:
            # Cargar imagen
            img_path = os.path.join(self.root, "images", self.imgs[idx])
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"Error al abrir la imagen {img_path}: {e}")
                return None, None

            # Cargar anotación
            annot_filename = os.path.splitext(self.imgs[idx])[0] + ".json"
            annot_path = os.path.join(self.root, "annotations", annot_filename)
            try:
                with open(annot_path) as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Error al cargar la anotación {annot_path}: {e}")
                return None, None

            # Procesar cajas y etiquetas
            boxes = []
            labels = []
            try:
                for shape in data["shapes"]:
                    label_str = shape["label"]
                    if label_str in label_map:
                        label = label_map[label_str]
                    else:
                        continue  # Se ignoran labels no definidos

                    pts = np.array(shape["points"])
                    xmin = np.min(pts[:, 0])
                    ymin = np.min(pts[:, 1])
                    xmax = np.max(pts[:, 0])
                    ymax = np.max(pts[:, 1])
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(label)
            except Exception as e:
                print(f"Error procesando los datos de {annot_path}: {e}")
                return None, None

            # Convertir a tensores y construir target
            try:
                boxes = torch.as_tensor(boxes, dtype=torch.float32)
                labels = torch.as_tensor(labels, dtype=torch.int64)
                image_id = torch.tensor([idx])
                area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
                iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
                target = {
                    "boxes": boxes,
                    "labels": labels,
                    "image_id": image_id,
                    "area": area,
                    "iscrowd": iscrowd
                }
            except Exception as e:
                print(f"Error creando el target para {img_path}: {e}")
                return None, None

            # ---- Aplicar redimensión ----
            # Definir la nueva resolución deseada (ancho, alto)
            new_size = (800, 600)
            # Obtener la resolución original
            orig_width, orig_height = img.size
            # Redimensionar la imagen
            img = img.resize(new_size)
            # Calcular los factores de escala
            scale_x = new_size[0] / orig_width
            scale_y = new_size[1] / orig_height
            # Ajustar las cajas de anotación
            target["boxes"] = target["boxes"] * torch.tensor([scale_x, scale_y, scale_x, scale_y])
            target["area"] = (target["boxes"][:, 3] - target["boxes"][:, 1]) * (target["boxes"][:, 2] - target["boxes"][:, 0])
            # ------------------------------

            # Aplicar transformaciones adicionales
            if self.transforms is not None:
                try:
                    # Es importante aplicar las transformaciones **después** de la redimensión
                    img = self.transforms(img)
                except Exception as e:
                    print(f"Error aplicando transformaciones a {img_path}: {e}")
                    return None, None

            return img, target

        except Exception as e:
            print(f"Error en __getitem__ para índice {idx}: {e}")
            return None, None

    def __len__(self):
        return len(self.imgs)

# Función para combinar lotes (necesaria para detección)
def collate_fn(batch):
    # Filtramos muestras nulas en caso de error
    batch = [b for b in batch if b[0] is not None and b[1] is not None]
    return tuple(zip(*batch))

# Transformaciones de imagen
def get_transform(train):
    transforms = [T.ToTensor()]
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

# --- 2. Modelo de Detección (Faster R-CNN) ---
def get_model(num_classes):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)

    #Reempla cabeza de clasificación para que se adapte a nuestro número de clases
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# --- 3. Función de Entrenamiento por Época ---
def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10):
    model.train()
    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=warmup_factor, total_iters=warmup_iters)
    
    for i, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        if lr_scheduler is not None:
            lr_scheduler.step()
        
        if i % print_freq == 0:
            print(f"Epoch {epoch} Iteration {i}, loss: {losses.item():.4f}")

# --- 4. Bucle Principal de Entrenamiento ---
def training_main():
    # Ruta raíz de tu dataset de entrenamiento
    dataset_root = "C:/VND_AI/data/no_labels"  # Asegurarse de tener subcarpetas 'images' y 'annotations' para evitar errores
    
    dataset = WindTurbineDamageDataset(dataset_root, transforms=get_transform(train=True))
    dataset_test = WindTurbineDamageDataset(dataset_root, transforms=get_transform(train=False))
    
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=1, collate_fn=collate_fn)
    
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=1, collate_fn=collate_fn)
    
    # Forzamos la ejecución en CPU
    device = torch.device('cpu')
    model = get_model(num_classes)
    model.to(device)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    num_epochs = 10
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        print(f"Finalizada la época {epoch}")
    
    torch.save(model.state_dict(), "fasterrcnn_wind_turbine_damage.pth")
    print("Modelo guardado.")

# --- 5. Función de Inferencia y Guardado de Imágenes Procesadas ---
def run_inference(model, device, input_root, output_root, detection_threshold=0.8):  #se puede cambiar el threshold para que detecte mas o menos daños
    """
    Recorre recursivamente las imágenes en input_root, ejecuta el modelo para detectar daños,
    y si se encuentran detecciones (con score > detection_threshold), se dibujan las cajas y se
    guarda la imagen en output_root, manteniendo la estructura de carpetas.
    """
    model.eval()
    transform = T.Compose([T.ToTensor()])
    
    for root, dirs, files in os.walk(input_root):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_image_path = os.path.join(root, file)
                try:
                    img = Image.open(input_image_path).convert("RGB")
                except Exception as e:
                    print(f"Error abriendo {input_image_path}: {e}")
                    continue
                
                image_tensor = transform(img).to(device)
                with torch.no_grad():
                    prediction = model([image_tensor])[0]
                
                # Filtrar predicciones por umbral de score
                boxes = prediction['boxes'][prediction['scores'] > detection_threshold]
                labels = prediction['labels'][prediction['scores'] > detection_threshold]
                scores = prediction['scores'][prediction['scores'] > detection_threshold]
                
                if boxes.nelement() == 0:
                    # No se detectó ningún daño con score alto
                    continue
                
                # Convertir imagen a formato OpenCV (BGR)
                cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                for box, label, score in zip(boxes, labels, scores):
                    box = box.int().cpu().numpy()
                    # Dibujar la caja y etiqueta
                    cv2.rectangle(cv_img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                    text = f"{label_names.get(int(label), 'N/A')} {score:.2f}"
                    cv2.putText(cv_img, text, (box[0], max(box[1]-10, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Construir la ruta de salida manteniendo la estructura de carpetas relativa
                relative_path = os.path.relpath(root, input_root)
                output_folder = os.path.join(output_root, relative_path)
                os.makedirs(output_folder, exist_ok=True)
                output_image_path = os.path.join(output_folder, file)
                cv2.imwrite(output_image_path, cv_img)
                print(f"Imagen procesada y guardada: {output_image_path}")

# --- 6. Ejecución ---
if __name__ == "__main__":
    # Para entrenar el modelo, descomenta la siguiente línea:
    # training_main()
    
    # Para la inferencia, se asume que ya tienes un modelo entrenado.
    # Carga el modelo entrenado:
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') solo si existe gpu disponible
     device = torch.device('cpu')
     model = get_model(num_classes)
     model_path = "fasterrcnn_wind_turbine_damage.pth"
     if os.path.exists(model_path):
         model.load_state_dict(torch.load(model_path, map_location=device))
         print("Modelo cargado.")
     else:
         print("No se encontró el modelo entrenado. Entrena primero el modelo.")
         exit(1)
    
     model.to(device)
    
    # Definir la ruta de la inspección con la estructura original y la ruta de salida
     inspection_input_root = "C:/VND_AI/data/test"  # Carpeta con las 3 subcarpetas de las palas para la parte de analisis, esto debe cambiar cada que acabe de analizar una pala
     output_root = os.path.join(inspection_input_root, "image_output")
    
     run_inference(model, device, inspection_input_root, output_root, detection_threshold=0.8)
