import cv2
import mediapipe as mp

# configuracao da MediaPipe para deteccao de landmarks
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
mp_drawing = mp.solutions.drawing_utils

# indices dos pontos que eu nao quero que sejam desenhados na imagem
landmarks_excluidas = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 18, 20, 22, 15, 17, 19, 21, 27, 28, 29, 30, 31, 32}


image_path = "C:/Users/jhona/OneDrive/Imagens/Screenshots/Captura de tela 2025-01-21 195619.png"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# processar a imagem para obter as landmarks
results = pose.process(image_rgb)

# se as landmarks foram detectadas
if results.pose_landmarks:
    annotated_image = image.copy()  

    # desenhando somente os pontos desejados
    for idx, landmark in enumerate(results.pose_landmarks.landmark):
        if idx not in landmarks_excluidas:
            # convertendo a posicao normalizada para coordenadas de pixel
            h, w, _ = image.shape
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(annotated_image, (x, y), 5, (0, 255, 0), -1)

    # exibindo a imagem com os pontos desenhados
    cv2.imshow("Landmarks Selecionadas", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Nenhuma landmark detectada na imagem.")
