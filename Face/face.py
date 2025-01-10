import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# configuraçoes de desenho para os pontos e conexoes
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))
connection_spec = mp_drawing.DrawingSpec(thickness=1, color=(255, 0, 0))

# cpaturando o video
video = cv2.VideoCapture(0)

if not video.isOpened():
    print("Erro ao abrir a camera!")
    exit()

# inicializando o FaceMesh com confiabilidade minima
with mp_face_mesh.FaceMesh(min_detection_confidence=0.5,
                           min_tracking_confidence=0.5) as face_mesh:

    while True:
        ret, image = video.read()
        if not ret:
            print("Erro ao capturar a imagem da camera!")
            break

        # convertendo para RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # processando a imagem para detectar faces
        results = face_mesh.process(image)

        # convertendo de volta para BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # desenhando as landmarks, se houver detecção
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=connection_spec
                )

        # exibindo a imagem com a malha facial
        cv2.imshow("Face Mesh", image)

        # Finalizando com a tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# liberando recursos
video.release()
cv2.destroyAllWindows()
