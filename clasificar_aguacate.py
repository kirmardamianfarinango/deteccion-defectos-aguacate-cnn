import cv2
from ultralytics import YOLO
import time

# =========================================================
# CONFIGURACIÓN GENERAL DEL SISTEMA
# =========================================================
# Ruta del modelo entrenado YOLOv8 para clasificación
MODEL_PATH = "/home/kirmar/models/best.pt"

# ID de la cámara USB (normalmente 0)
CAM_ID = 0

# Tamaño de imagen usado en entrenamiento del modelo
IMG_SIZE = 224

# Nombre de la clase considerada como producto aceptable
OK_NAME = "OK"


def main():
    # Carga del modelo de clasificación entrenado
    print("Cargando modelo:", MODEL_PATH)
    model = YOLO(MODEL_PATH)
    print("Modelo cargado correctamente")

    # Inicialización de la cámara USB
    cap = cv2.VideoCapture(CAM_ID)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara USB")

    # Configuración básica de resolución y velocidad para mejorar fluidez
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Variable para cálculo de FPS
    prev_t = time.time()

    # Bucle principal de adquisición e inferencia en tiempo real
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("No se pudo capturar el frame")
            break

        # Ejecución de inferencia con el modelo YOLOv8 (clasificación)
        results = model.predict(
            source=frame,
            imgsz=IMG_SIZE,
            verbose=False
        )

        # Obtención del resultado principal
        r = results[0]
        probs = r.probs

        # Clase con mayor probabilidad y su nivel de confianza
        cls_id = int(probs.top1)
        conf = float(probs.top1conf)
        cls_name = model.names[cls_id]

        # Regla de decisión: BUENO / MALO
        if cls_name == OK_NAME:
            estado = "AGUACATE BUENO"
            color = (0, 255, 0)
        else:
            estado = "AGUACATE MALO"
            color = (0, 0, 255)

        # Cálculo de cuadros por segundo (FPS)
        now = time.time()
        fps = 1.0 / max(now - prev_t, 1e-6)
        prev_t = now

        # Visualización de resultados sobre la imagen
        texto_estado = f"{estado} ({cls_name}) conf={conf:.2f}"
        texto_fps = f"FPS: {fps:.1f}"

        cv2.putText(frame, texto_estado, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(frame, texto_fps, (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Visualización en ventana
        cv2.imshow("Clasificador Aguacate (6 clases -> BUENO/MALO)", frame)

        # Salida controlada del sistema
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Liberación de recursos
    cap.release()
    cv2.destroyAllWindows()


# Punto de entrada principal del programa
if __name__ == "__main__":
    main()
