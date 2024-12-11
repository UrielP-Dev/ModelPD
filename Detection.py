import cv2
from PIL import Image
import numpy as np
import supervision as sv
from inference import get_model

def convert_frame_to_image(frame):
    """
    Convierte un frame de OpenCV (BGR) a una imagen PIL (RGB).
    """
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

# Inicializa la captura de video (0 para la cámara predeterminada)
video_capture = cv2.VideoCapture(0)  # Cambia el índice si tienes múltiples cámaras

# Carga un modelo YOLOv8 preentrenado
model = get_model(model_id="yolov8n-640")

# Crea anotadores para las detecciones
bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

while True:
    # Captura un frame del video
    ret, frame = video_capture.read()
    if not ret:
        print("Error al capturar el video. Saliendo...")
        break

    # Convierte el frame a una imagen PIL
    image = convert_frame_to_image(frame)

    # Realiza inferencias en el frame
    results = model.infer(image)[0]

    # Convierte los resultados en detecciones para supervisión
    detections = sv.Detections.from_inference(results)

    # Genera etiquetas para las detecciones
    labels = [
        f"ID:{idx} {detections.data['class_name'][idx]} {detections.confidence[idx]:.2%}"
        for idx in range(len(detections.data['class_name']))
    ]

    # Anota el frame con las cajas de detección y etiquetas
    annotated_frame = bounding_box_annotator.annotate(scene=image, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

    # Convierte la imagen PIL anotada de regreso a un array de OpenCV (BGR)
    annotated_frame = cv2.cvtColor(np.array(annotated_frame), cv2.COLOR_RGB2BGR)

    # Muestra el frame anotado
    cv2.imshow("Detección en tiempo real", annotated_frame)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera la captura de video y cierra las ventanas
video_capture.release()
cv2.destroyAllWindows()
