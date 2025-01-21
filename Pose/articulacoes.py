import cv2
import mediapipe as mp

# inicializando o mediapipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

image = cv2.imread("C:/Users/jhona/OneDrive/Imagens/Screenshots/Captura de tela 2025-01-21 195619.png")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# processando a imagem
results = pose.process(image_rgb)

# verificando se pontos do corpo foram detectados
if results.pose_landmarks:
    for idx, landmark in enumerate(results.pose_landmarks.landmark):
        print(f"Point {idx}: ({landmark.x}, {landmark.y}, {landmark.z})")
else:
    print("Nenhum ponto detectado.")

# desenhando os pontos na imagem
mp_drawing = mp.solutions.drawing_utils
mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

# mostrando a imagem com os pontos desenhados
cv2.imshow("Pose Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
