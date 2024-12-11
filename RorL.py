from ultralytics import YOLO
import cv2

def detect_movement_with_yolo():
    # Carga el modelo preentrenado de YOLO
    model = YOLO('yolov8n.pt')

    # Inicia la cámara
    cap = cv2.VideoCapture(0)

    # Variables para rastrear movimiento
    prev_x = None
    direction = "Estático"  # Inicialización predeterminada

    print("Presiona 'q' para salir.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error al acceder a la cámara.")
                break

            # Realiza la predicción usando YOLO
            results = model(frame, conf=0.5)  # Confianza mínima del 50%

            # Filtra detecciones de personas
            for result in results[0].boxes:
                class_id = int(result.cls[0])  # Clase detectada
                if class_id == 0:  # Clase 0 es 'person' en el modelo COCO
                    x1, y1, x2, y2 = map(int, result.xyxy[0])  # Coordenadas del cuadro delimitador
                    current_x = (x1 + x2) // 2  # Coordenada X central de la persona detectada

                    # Determina la dirección del movimiento
                    if prev_x is not None:
                        if current_x > prev_x + 10:
                            direction = "Derecha"
                        elif current_x < prev_x - 10:
                            direction = "Izquierda"
                        else:
                            direction = "Estático"
                        print(f"Moviendo: {direction}")
                    
                    prev_x = current_x

                    # Dibuja el cuadro delimitador y la dirección
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, direction, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Muestra el video con detecciones
            cv2.imshow('Detección de movimiento con YOLO', frame)

            # Presiona 'q' para salir
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Finalizando...")

    # Libera la cámara y destruye las ventanas
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_movement_with_yolo()
