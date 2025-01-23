import cv2
import mediapipe as mp


mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
mp_drawing = mp.solutions.drawing_utils

excluded_landmarks = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 18, 20, 22, 15, 17, 19, 21, 27, 28, 29, 30, 31, 32}

image_path = "C:/Users/jhona/OneDrive/Imagens/Screenshots/Captura de tela 2025-01-21 195619.png"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

results = pose.process(image_rgb)

if results.pose_landmarks:
    annotated_image = image.copy()

    # obtendo conexoes (pares de pontos) do modelo MediaPipe
    pose_connections = mp_pose.POSE_CONNECTIONS

    # desenhando somente os pontos de interesse
    for idx, landmark in enumerate(results.pose_landmarks.landmark):
        if idx not in excluded_landmarks:
            h, w, _ = image.shape
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(annotated_image, (x, y), 5, (0, 255, 0), -1)

    # desenhando somente as linhas de interesse
    for connection in pose_connections:
        start_idx, end_idx = connection
        if start_idx not in excluded_landmarks and end_idx not in excluded_landmarks:
            # coordenadas do ponto inicial
            start_landmark = results.pose_landmarks.landmark[start_idx]
            x1, y1 = int(start_landmark.x * w), int(start_landmark.y * h)

            # coordenadas do ponto final
            end_landmark = results.pose_landmarks.landmark[end_idx]
            x2, y2 = int(end_landmark.x * w), int(end_landmark.y * h)

            # desenhando a linha
            cv2.line(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # exibindo a imagem com os pontos e linhas desenhados
    cv2.imshow("Landmarks com Linhas", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Nenhuma landmark detectada na imagem.")
