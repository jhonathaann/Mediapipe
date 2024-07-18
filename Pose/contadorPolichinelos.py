import cv2
import mediapipe as mp
import math

# abrindo o vídeo
video = cv2.VideoCapture('C:/Users/jhona/OneDrive/Projetos/Mediapipe/Pose/polichinelos.mp4')

# criando as variaveis para a detecção dos postos do corpo
pose = mp.solutions.pose
pose_detection = pose.Pose(min_tracking_confidence=0.5,min_detection_confidence=0.5)
# esse parametros servem para melhorar a detecção do video
draw = mp.solutions.drawing_utils # variavel que irá desenhar as linhas dos pontos dentro do video

# para as maões eu irei selecionar os pontos 20 (right_index) e 19 (left_index)
# para os pés eu irei selecionar os pontos 32(right_foot_inedex) e 31 (left_foot_index)

contador = 0
flag = True

while True:
    # para a leitura do video precisamos de duas variaveis. a primeira delas é para checar se a imagem esta rodando
    check, img = video.read()

    
    # convertendo essa imagem para RGB
    videoRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    # criando uma variavel que ira receber o resultado da detecção
    results = pose_detection.process(videoRGB)


    # criando uma variavel que ira extrair de results os pontos em coordenadas do corpo
    points = results.pose_landmarks

    # agora vamos desenhar esses pontos dentro do video. esse ultimo parametro representa o tipo de desenho que nos queremos
    draw.draw_landmarks(img, points, pose.POSE_CONNECTIONS)

    # extraindo as dimensões do vídeo
    altura, largura,_ = img.shape

    # agora, vamos ensinar ao computador o que é um polichinelo
    # primeiro, vamos verificar se a variavel points não esta vazia
    if points:
        #print(points)

        # nos nao precisamos de todos os pontos da imagem. precisamos apenas do 19,20,31,32
        pe_direitoy = float(points.landmark[pose.PoseLandmark.RIGHT_FOOT_INDEX].y) # cada landmark possui 3 coordenadas (.y para pegar o y)
        pe_direitox = float(points.landmark[pose.PoseLandmark.RIGHT_FOOT_INDEX].x)

        pe_esquerdoy = float(points.landmark[pose.PoseLandmark.LEFT_FOOT_INDEX].y) 
        pe_esquerdox = float(points.landmark[pose.PoseLandmark.LEFT_FOOT_INDEX].x)

        mao_direitay = float(points.landmark[pose.PoseLandmark.RIGHT_INDEX].y) 
        mao_direitax = float(points.landmark[pose.PoseLandmark.RIGHT_INDEX].x)

        mao_esquerday = float(points.landmark[pose.PoseLandmark.LEFT_INDEX].y) 
        mao_esquerdax = float(points.landmark[pose.PoseLandmark.LEFT_INDEX].x)

        # precisamos fazer uma conversão desses valores para pixeis, para isso, devemos multiplicar a altura (em y) e o comprimento em (x)
        
        pe_direitoy = pe_direitoy * altura
        pe_direitox = pe_direitox *  largura


        pe_esquerdoy = pe_esquerdoy * altura
        pe_esquerdox = pe_esquerdox * largura

        mao_direitay = mao_direitay * altura
        mao_direitax = mao_direitax * largura

        mao_esquerday = mao_esquerday * altura
        mao_esquerdax = mao_esquerdax * largura

        # medindo as distancia esntre esses pontos
        dinst_maos = math.hypot(mao_direitax - mao_esquerdax, mao_direitay - mao_esquerday)
        disnt_pes = math.hypot(pe_direitox - pe_esquerdox, pe_direitoy - pe_esquerdoy)

        # maos < 150 e pes > 150. isso sera um polichinelo
        if flag == True and dinst_maos <= 150 and disnt_pes >= 150:
            contador+=1
            flag = False
        
        # criando uma situação oposta, para que ele contabilize não a cada frame, mas sim a cada polichinelo
        if dinst_maos > 150 and disnt_pes < 150      :
            flag = True

        print(f"Distância entre as maões: {dinst_maos}, Dinstância entre os pés: {disnt_pes}")

        #print(contador)

        # colocando o contador no video
        texto = f'QTD {contador}'
        cv2.rectangle(img,(20,240),(280,120),(255,0,0),-1)
        cv2.putText(img,texto,(40,200),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),5)

    # mostrando o vídeo
    cv2.imshow('Resultado',img)
    cv2.waitKey(40) # 40 mili segundos de delay

