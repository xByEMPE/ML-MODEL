import os
import io
import json
import numpy as np
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
import torchvision.transforms as T
from PIL import Image
from supabase import create_client
from datetime import datetime, timezone
import logging

# --- CONFIGURACIÓN DE SUPABASE ---
SB_URL = "https://yhmhpxwnxcbpouyoxarr.supabase.co"
SB_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlobWhweHdueGNicG91eW94YXJyIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3MTMzNTI1NCwiZXhwIjoyMDg2OTExMjU0fQ.1zPCct6TN3zWegnUolgRNk3XH1YT9OKqq8rwFU_Sx2Y"
BUCKET = "images"

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("inference.log"),
        logging.StreamHandler()
    ]
)

# Inicializar cliente Supabase
sb = create_client(SB_URL, SB_KEY)

# --- CONFIGURACIÓN DE ETIQUETAS ---
label_map = {
    "Categoria 1": 1,
    "Categoria 2": 2,
    "Categoria 3": 3,
    "Categoria 4": 4,
    "Categoria 5": 5
}
# Mapeo inverso
label_names = {v: k for k, v in label_map.items()}
num_classes = len(label_map) + 1  # +1 para fondo

# --- MODELO (Solo Inferencia) ---
def get_model(num_classes):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# --- FUNCIONES AUXILIARES SUPABASE ---
def get_captures_to_process():
    """
    Obtiene 'captures' que aún no han sido procesadas (no tienen findings generados por AI).
    Para evitar re-procesamiento, revisamos si existe algún finding asociado a este capture_id
    que tenga ai_metadata -> 'model' = 'fasterrcnn'.
    """
    try:
        # 1. Obtener todas las capturas con la información relacional necesaria
        #    Corregido para evitar 'duplicate_alias': anidar relaciones en un solo bloque
        
        all_captures = []
        limit = 1000
        offset = 0
        
        while True:
            # Quitamos el filtro puro de BD porque las inspecciones antiguas
            # no tienen la llave 'ai_processed' en el JSON, lo que causaba
            # que se bajaran creyendo que no estaban procesadas.
            response = sb.table("captures").select("""
                id, storage_path, section, blade_id, raw_metadata,
                blades!inner(
                    position,
                    turbine_id,
                    turbines!inner(
                        id,
                        turbine_name,
                        project_id,
                        projects!inner(
                            name,
                            wind_farm
                        )
                    )
                )
            """).range(offset, offset + limit - 1).execute()
            
            data = response.data if response.data else []
            all_captures.extend(data)
            
            if len(data) < limit:
                break
                
            offset += limit
            
        # 2. Filtrar las que ya tienen findings de este modelo (Capturas Legacy antes de usar ai_processed)
        if not all_captures:
            return []
            
        captures_to_check_legacy = []
        valid_captures = []
        
        for cap in all_captures:
            raw_data = cap.get('raw_metadata') or {}
            
            # Comprobación de emergencia (Legacy) para proyectos viejos que ya fueron inferidos
            # Ya sea con la etiqueta original booleana, el string, o las llaves de timestamp.
            is_processed = (
                raw_data.get('ai_processed') is True or 
                str(raw_data.get('ai_processed')).lower() == 'true' or 
                'ai_processed_at' in raw_data or 
                'ai_processed_by' in raw_data
            )
            
            if is_processed:
                continue
                
            captures_to_check_legacy.append(cap)
            
        # Obtener todos los findings de IA existentes (con paginacion)
        processed_ids = set()
        limit_f = 1000
        offset_f = 0
        while True:
            res = sb.table("findings").select("capture_id").not_.is_("ai_metadata", "null").range(offset_f, offset_f + limit_f - 1).execute()
            data_f = res.data if res.data else []
            processed_ids.update(f['capture_id'] for f in data_f)
            if len(data_f) < limit_f:
                break
            offset_f += limit_f
            
        for cap in captures_to_check_legacy:
            if cap['id'] not in processed_ids:
                valid_captures.append(cap)
        
        logging.info(f"Total capturas: {len(all_captures)}. A procesar: {len(valid_captures)}")
        return valid_captures

    except Exception as e:
        logging.error(f"Error obteniendo captures: {e}")
        return []

def download_image_from_supabase(storage_path):
    try:
        data = sb.storage.from_(BUCKET).download(storage_path)
        return data
    except Exception as e:
        logging.error(f"Error descargando imagen {storage_path}: {e}")
        return None

def upload_json_to_supabase(json_path, json_data):
    try:
        # Verificar existencia antes de subir (segunda capa de anti-duplicados)
        try:
            exists = sb.storage.from_(BUCKET).list(os.path.dirname(json_path))
            file_name = os.path.basename(json_path)
            if any(f['name'] == file_name for f in exists):
                logging.info(f"JSON ya existe en storage: {json_path}")
                return True # Asumimos éxito si ya existe
        except:
            pass

        json_bytes = json.dumps(json_data, ensure_ascii=False).encode('utf-8')
        result = sb.storage.from_(BUCKET).upload(
            json_path, 
            json_bytes, 
            {"contentType": "application/json", "upsert": "true"}
        )
        if result:
            logging.info(f"JSON subido: {json_path}")
            return True
        return False
    except Exception as e:
        logging.error(f"Error subiendo JSON {json_path}: {e}")
        return False

def create_processed_json_path(original_path, project_name, wind_farm, turbine_name, blade_name, section):
    file_name = os.path.splitext(os.path.basename(original_path))[0]
    return f"processed/{project_name}/{wind_farm}/{turbine_name}/{blade_name}/{section}/{file_name}.json"

def save_findings_to_database(detections, capture_id):
    """Guarda los daños detectados en la tabla 'findings'. Si no hay daños, retorna True sin insertar."""
    try:
        if not detections:
            # Ya no insertamos un "ai_log" falso porque la tabla de findings lo interpreta como un daño real.
            # La deduplicación ahora la maneja la columna 'raw_metadata' de la tabla 'captures' directamente.
            return True
            
        now = datetime.now(timezone.utc).isoformat()
        findings_records = []
        
        for detection in detections:
                # Preparar datos para las columnas jsonb
                
                # 'region' contendrá la geometria normalizada y absoluta
                region_data = {
                    "bbox": detection.get("bbox", []),                # [x1, y1, x2, y2] px
                    "bbox_xywh_norm": detection.get("bbox_xywh_norm", []), # [x, y, w, h] 0-1 (CRITICO para frontend)
                    "image_dims": {
                        "width": detection.get("image_width"),
                        "height": detection.get("image_height")
                    }
                }
                
                # 'attributes' contendrá detalles del daño
                attributes_data = {
                    "category": detection.get("category", "Daño"),
                    "area_px": detection.get("area", 0)
                }
                
                # 'ai_metadata' para rastrear origen
                ai_data = {
                    "model": "fasterrcnn",
                    "version": "1.0",
                    "confidence": detection.get("confidence", 0.0),
                    "processed_at": now
                }
                
                finding_record = {
                    "capture_id": capture_id,
                    "type": "damage",
                    "severity": detection.get("severity", "Cat 1"), # Mapeo directo a columna severity
                    "confidence": detection.get("confidence", 0.0),
                    "region": region_data,
                    "attributes": attributes_data,
                    "ai_metadata": ai_data,
                    "created_at": now,
                    "updated_at": now
                }
                
                findings_records.append(finding_record)
        
        # Insertar en tabla 'findings'
        if findings_records:
            result = sb.table("findings").insert(findings_records).execute()
            if result.data:
                logging.info(f"Insertados {len(findings_records)} findings en BD")
                return True
            else:
                logging.error("Error insertando findings")
                return False
        return True
            
    except Exception as e:
        logging.error(f"Error guardando findings en BD: {e}")
        return False

# --- PROCESO PRINCIPAL DE INFERENCIA ---
def run_inference_supabase(model, device, detection_threshold=0.4):
    model.eval()
    transform = T.Compose([T.ToTensor()])
    
    captures = get_captures_to_process()
    if not captures:
        logging.info("No hay nuevas capturas para procesar.")
        return
    
    logging.info(f"Iniciando procesamiento de {len(captures)} capturas...")
    
    processed_count = 0
    errors = 0
    processed_turbines_ids = set()
    
    for cap in captures:
        try:
            # Extraer metadatos relacionales
            capture_id = cap['id']
            file_path = cap['storage_path']
            section = cap['section']
            
            # Datos de relaciones (nested)
            # cap['blades'] es un objeto (singular) porque es !inner join de relación
            blade_obj = cap.get('blades', {})
            turbine_obj = blade_obj.get('turbines', {})
            project_obj = turbine_obj.get('projects', {})
            
            project_name = project_obj.get('name', 'UnknownProject')
            wind_farm = project_obj.get('wind_farm', 'UnknownFarm')
            turbine_name = turbine_obj.get('turbine_name', 'UnknownTurbine')
            
            # Mapeo de Blade Position a Nombre ("A" -> "Pala A" o consistente con frontend)
            # El frontend usa "Blade A", "Blade B". El usuario mencionó "Pala A".
            # Vamos a estandarizar a lo que usa el sistema de carpetas y frontend.
            pos = blade_obj.get('position', 'A') # Asumimos 'A', 'B', 'C'
            blade_name_display = f"Blade {pos}" if len(pos) == 1 else pos
            # Para carpetas, a veces se prefiere 'Pala'. Mantengamos lógica previa de carpetas:
            folder_blade_name = f"Pala {pos}" if len(pos) == 1 else pos
            
            # Ruta destino JSON (Storage)
            processed_json_path = create_processed_json_path(
                file_path, project_name, wind_farm, turbine_name, folder_blade_name, section
            )
            
            # 1. Descargar imagen
            image_data = download_image_from_supabase(file_path)
            if not image_data:
                errors += 1
                continue
            
            img = Image.open(io.BytesIO(image_data)).convert("RGB")
            orig_width, orig_height = img.size
            
            # 2. Preprocesamiento
            target_size = (800, 600)
            img_resized = img.resize(target_size)
            
            # 3. Inferencia
            image_tensor = transform(img_resized).to(device)
            with torch.no_grad():
                prediction = model([image_tensor])[0]
            
            # 4. Procesar Detecciones
            boxes = prediction['boxes'][prediction['scores'] > detection_threshold]
            labels = prediction['labels'][prediction['scores'] > detection_threshold]
            scores = prediction['scores'][prediction['scores'] > detection_threshold]
            
            detections = []
            
            if len(boxes) > 0:
                boxes = boxes.cpu().numpy()
                labels = labels.cpu().numpy()
                scores = scores.cpu().numpy()
                
                scale_x = orig_width / target_size[0]
                scale_y = orig_height / target_size[1]
                
                for box, label, score in zip(boxes, labels, scores):
                    x1_pred, y1_pred, x2_pred, y2_pred = box
                    
                    # Convertir a float nativo explícitamente y calcular escalas
                    x1_orig = float(max(0, min(orig_width, x1_pred * scale_x)))
                    y1_orig = float(max(0, min(orig_height, y1_pred * scale_y)))
                    x2_orig = float(max(0, min(orig_width, x2_pred * scale_x)))
                    y2_orig = float(max(0, min(orig_height, y2_pred * scale_y)))
                    
                    w_orig = x2_orig - x1_orig
                    h_orig = y2_orig - y1_orig
                    
                    if w_orig <= 0 or h_orig <= 0: continue
                    
                    # Normalización (asegurar floats nativos)
                    bbox_xywh_norm = [
                        round(float(x1_orig / orig_width), 6),
                        round(float(y1_orig / orig_height), 6),
                        round(float(w_orig / orig_width), 6),
                        round(float(h_orig / orig_height), 6)
                    ]
                    
                    label_int = int(label)
                    category = label_names.get(label_int, f"Categoria {label_int}")
                    # BD espera enum "1", "2", etc. No "Cat 1".
                    # Frontend parsea ints de cualquier string, así que "1" funciona.
                    severity = str(label_int)
                    
                    detections.append({
                        "category": category,
                        "severity": severity,
                        "confidence": round(float(score), 4),
                        "bbox": [round(x1_orig, 2), round(y1_orig, 2), round(x2_orig, 2), round(y2_orig, 2)],
                        "bbox_xywh": [round(x1_orig, 2), round(y1_orig, 2), round(w_orig, 2), round(h_orig, 2)],
                        "bbox_xywh_norm": bbox_xywh_norm,
                        "image_width": int(orig_width),
                        "image_height": int(orig_height),
                        "area": round(float(w_orig * h_orig), 2)
                    })

            # 5. Generar Payload JSON para Storage (Legacy/Backup usage)
            now = datetime.now(timezone.utc).isoformat()
            payload = {
                "project": project_name,
                "wind_farm": wind_farm,
                "turbine": turbine_name,
                "blade": blade_name_display,
                "section": section,
                "source_path": file_path,
                "processed_at": now,
                "detections": detections,
                "file_info": {
                    "width": orig_width,
                    "height": orig_height,
                    "original_path": file_path
                }
            }
            
            # 6. Subir JSON (Backup/Bucket)
            # Solo subimos si logramos procesar la imagen correctamente
            upload_json_to_supabase(processed_json_path, payload)
                
            # 7. Actualizar Base de Datos (Findings)
            if save_findings_to_database(detections, capture_id):
                # FIRMA A PRUEBA DE BALAS:
                # Actualizar permanentemente el registro de "captures" indicando que ya pasó por la IA.
                # Así, aunque los usuarios manipulen o eliminen hallazgos, jamás se reprocesará la imagen.
                now_str = datetime.now(timezone.utc).isoformat()
                
                # Rescatar el metadata actual para no sobreescribir otros datos y anñadir la bandera
                current_raw = cap.get('raw_metadata') or {}
                current_raw['ai_processed'] = True
                current_raw['ai_processed_by'] = 'fasterrcnn'
                current_raw['ai_processed_at'] = now_str

                sb.table("captures").update({
                    "raw_metadata": current_raw
                }).eq("id", capture_id).execute()

                processed_count += 1
                
                # Guardar el turbine_id exitoso para notificar al final
                tid = turbine_obj.get('id')
                if tid:
                    processed_turbines_ids.add(tid)
            else:
                errors += 1

        except Exception as e:
            logging.error(f"Error procesando capture {cap.get('id')}: {e}")
            errors += 1

    logging.info(f"Fin del proceso. Procesados: {processed_count}, Errores: {errors}")
    
    # --- EDGE FUNCTION TRIGGER ---
    # Notificar por Email a Admins sobre las turbinas recién procesadas
    if processed_turbines_ids:
        logging.info(f"Enviando notificaciones de correo para {len(processed_turbines_ids)} turbina(s)...")
        for tid in processed_turbines_ids:
            try:
                # El edge function espera body: { "turbineId": <int> } 
                res = sb.functions.invoke('notify-inspection-complete', invoke_options={'body': {'turbineId': tid}})
                logging.info(f"Notificación de turbina ID {tid} enviada. Respuesta: {res}")
            except Exception as e:
                logging.error(f"Error invocando la Edge Function para turbina ID {tid}: {e}")
    # -----------------------------

if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Usando dispositivo: {device}")
    
    # Cargar modelo
    model = get_model(num_classes)
    model_path = "fasterrcnn_wind_turbine_damage.pth"
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Modelo cargado.")
        model.to(device)
        
        # Ejecutar
        run_inference_supabase(model, device)
    else:
        print(f"Error: No se encontró {model_path}")
