import cv2
from ultralytics import YOLO
import numpy as np

class PersonTracker:
    def __init__(self, video_path):
        self.model = YOLO('yolov8n.pt')
        self.cap = cv2.VideoCapture(video_path)
        
        # Inicializar variables de seguimiento
        self.tracks = {}
        self.next_track_id = 0
        
        # Umbrales de movimiento y eliminación
        self.movement_threshold = 20  # Píxeles para considerar movimiento significativo
        self.max_frames_offscreen = 30  # Número de frames para considerar un track como eliminable
        
    def calculate_trajectory(self, prev_bbox, current_bbox):
        """
        Calcula la dirección del movimiento entre dos posiciones de bounding box
        """
        prev_center = ((prev_bbox[0] + prev_bbox[2]) / 2, 
                       (prev_bbox[1] + prev_bbox[3]) / 2)
        curr_center = ((current_bbox[0] + current_bbox[2]) / 2, 
                       (current_bbox[1] + current_bbox[3]) / 2)
        
        dx = curr_center[0] - prev_center[0]
        dy = curr_center[1] - prev_center[1]
        
        if abs(dx) > abs(dy) and abs(dx) > self.movement_threshold:
            return 'derecha' if dx > 0 else 'izquierda'
        elif abs(dy) > self.movement_threshold:
            return 'abajo' if dy > 0 else 'arriba'
        return None  # No se considera estático, sólo los movimientos significativos
    
    def is_bbox_in_frame(self, bbox, frame_shape):
        """
        Verifica si el bounding box está dentro de los límites del frame
        """
        return (bbox[0] >= 0 and bbox[2] <= frame_shape[1] and 
                bbox[1] >= 0 and bbox[3] <= frame_shape[0])
    
    def track_persons(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Detección con YOLO
            results = self.model(frame, conf=0.5)
            
            # Actualizar y eliminar tracks
            tracks_to_remove = []
            
            # Procesar cada track existente
            for track_id, track in list(self.tracks.items()):
                track['frames_offscreen'] = track.get('frames_offscreen', 0) + 1
                
                # Eliminar tracks que lleven muchos frames fuera de la pantalla
                if track['frames_offscreen'] > self.max_frames_offscreen:
                    tracks_to_remove.append(track_id)
            
            # Eliminar tracks marcados
            for track_id in tracks_to_remove:
                del self.tracks[track_id]
            
            # Procesar cada detección de persona
            current_detections = []
            for result in results[0].boxes:
                if int(result.cls[0]) == 0:  # Clase de persona
                    bbox = result.xyxy[0].cpu().numpy()
                    current_detections.append(bbox)
                    
                    # Encontrar o crear un track para esta persona
                    closest_track = self.find_closest_track(bbox)
                    
                    if closest_track is not None:
                        # Actualizar track existente
                        trajectory = self.calculate_trajectory(
                            self.tracks[closest_track]['last_bbox'], 
                            bbox
                        )
                        
                        if trajectory:  # Solo actualizamos si hubo un movimiento significativo
                            self.tracks[closest_track].update({
                                'last_bbox': bbox,
                                'last_trajectory': trajectory,
                                'frames_offscreen': 0  # Reiniciar contador de frames fuera de pantalla
                            })
                    else:
                        # Crear nuevo track sin etiqueta 'inicio'
                        self.tracks[self.next_track_id] = {
                            'last_bbox': bbox,
                            'last_trajectory': None,  # Sin etiqueta de inicio
                            'frames_offscreen': 0,
                            'first_seen': frame.copy()
                        }
                        self.next_track_id += 1
            
            # Visualización (opcional)
            self.visualize_tracks(frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()
    
    def find_closest_track(self, bbox, max_distance=100):
        """
        Encuentra el track más cercano a la nueva detección
        """
        bbox_center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        
        for track_id, track in self.tracks.items():
            last_bbox = track['last_bbox']
            last_center = ((last_bbox[0] + last_bbox[2]) / 2, 
                           (last_bbox[1] + last_bbox[3]) / 2)
            
            distance = np.sqrt(
                (bbox_center[0] - last_center[0])**2 + 
                (bbox_center[1] - last_center[1])**2
            )
            
            if distance < max_distance:
                return track_id
        
        return None
    
    def visualize_tracks(self, frame):
        """
        Visualiza los tracks y sus trayectorias
        """
        for track_id, track in list(self.tracks.items()):
            bbox = track['last_bbox']
            trajectory = track.get('last_trajectory', 'desconocido')
            
            # Dibujar bounding box
            cv2.rectangle(
                frame, 
                (int(bbox[0]), int(bbox[1])), 
                (int(bbox[2]), int(bbox[3])), 
                (0, 255, 0), 
                2
            )
            
            # Añadir texto de trayectoria
            cv2.putText(
                frame, 
                f"Track {track_id}: {trajectory}", 
                (int(bbox[0]), int(bbox[1]-10)), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 0, 255), 
                2
            )
        
        cv2.imshow('Tracking de Personas', frame)

if __name__ == "__main__":
    tracker = PersonTracker("Video8.mp4")
    tracker.track_persons()
 