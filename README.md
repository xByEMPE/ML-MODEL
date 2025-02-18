# Modelo de ML para la deteccion de daños en palas de aerogeneradores

### Detección de Daños en Turbinas Eólicas

Este repositorio contiene un proyecto basado en PyTorch y TorchVision para entrenar e implementar un modelo de detección de daños en imágenes de turbinas eólicas (o palas). Se utiliza el modelo **Faster R-CNN** con una arquitectura **ResNet50-FPN** adaptada a un conjunto de clases definidas por el usuario.

## Contenido

- **Dataset Personalizado:**  
  La clase `WindTurbineDamageDataset` se encarga de cargar las imágenes y sus correspondientes anotaciones (en formato JSON). Se espera que la estructura del dataset sea la siguiente:
  *dataset_root/ images/ # Imágenes en formato .png, .jpg o .jpeg annotations/ # Archivos JSON con las anotaciones (el mismo nombre base que la imagen)

  


- **Transformaciones y Preprocesamiento:**  
Se aplican transformaciones básicas utilizando `torchvision.transforms` para convertir las imágenes a tensores y, opcionalmente, realizar flip horizontal en el entrenamiento.

- **Modelo de Detección (Faster R-CNN):**  
El modelo se carga utilizando la nueva API de TorchVision. En la función `get_model`, se usan los pesos preentrenados:
```python
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

def get_model(num_classes):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model
```

Esto elimina los warnings asociados al uso del parámetro pretrained.

- **Entrenamiento:**
La función training_main() prepara el dataset y ejecuta el entrenamiento del modelo durante un número definido de épocas. Los pesos entrenados se guardan en el archivo fasterrcnn_wind_turbine_damage.pth.
Nota: Para usar solo inferencia, se debe comentar la línea de entrenamiento en el bloque principal.

- **Inferencia y Guardado de Resultados:**
La función run_inference() recorre recursivamente una carpeta de imágenes de entrada (inspection_input_root), realiza la detección y, si la puntuación de alguna detección supera el umbral (detection_threshold), dibuja la caja y la etiqueta sobre la imagen. Las imágenes procesadas se guardan en una carpeta image_output dentro de la carpeta de inspección, manteniendo la estructura de directorios original.

Ejemplo de filtrado de detecciones por umbral:
```
boxes = prediction['boxes'][prediction['scores'] > detection_threshold]
labels = prediction['labels'][prediction['scores'] > detection_threshold]
scores = prediction['scores'][prediction['scores'] > detection_threshold]
```


- **Configuración del Umbral de Detección:**

Puedes ajustar el umbral de detección modificando el parámetro detection_threshold en la función run_inference. Por ejemplo, se puede probar con un valor de 0.5 para aumentar la sensibilidad o modificarlo según los resultados obtenidos.

## Requisitos 
* Python 3.8 o superior
* PyTorch
* TorchVision (se recomienda la versión actualizada para utilizar la nueva API de pesos)
* OpenCV (opencv-python)
* Pillow
* NumPy

Puedes instalar las dependencias con:

```
pip install torch torchvision opencv-python Pillow numpy

```

## Uso 
- **Entrenamiento** 
Para entrenar el modelo:

1. Asegúrate de tener el dataset organizado correctamente en la carpeta definida en dataset_root (por defecto: C:/VND_AI/data/no_labels).
2. Descomenta la línea training_main() en el bloque principal del código.
3. Ejecuta el script:

```
python main.py

```

4. Los pesos se guardarán en fasterrcnn_wind_turbine_damage.pth.


- **Inferencia**

Para ejecutar la inferencia sobre nuevas imágenes:

1. Asegúrate de que el modelo ya se haya entrenado y que el archivo fasterrcnn_wind_turbine_damage.pth esté disponible.
2. Configura la variable inspection_input_root para apuntar a la carpeta de entrada (por ejemplo, data/test/A5.02) que contiene las subcarpetas con las imágenes a analizar.
3. Verifica que la línea de entrenamiento esté comentada para no repetir el proceso de entrenamiento.
4. Ejecuta el script: 

```
python main.py

```
5. Las imágenes procesadas se guardarán en image_output dentro de la carpeta de entrada.

## Notas adicionales
* Dispositivo:
Por defecto, el código fuerza el uso de la CPU (definido en device = torch.device('cpu')) para evitar problemas con la función NMS en CUDA. Si cuentas con GPU y deseas utilizarla, ajusta la selección del dispositivo y asegúrate de que tu instalación de TorchVision soporte NMS en CUDA.

* Estructura del Código:
El script está dividido en secciones:

- Definición del dataset y transformaciones.
- Funciones para obtener el modelo y modificar la cabeza de clasificación.
- Funciones de entrenamiento e inferencia.
- Bloque principal ```(if __name__ == "__main__":)``` para seleccionar entre entrenamiento o inferencia.