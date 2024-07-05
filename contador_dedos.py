# utilizando a solução hand da biblioteca midiapipe, que nós fornece o mapeamento das mãos
import cv2
import mediapipe as mp

# variavel que vai abrir a webcam
video = cv2.VideoCapture(0)  # 0 porque eu so tenho uma camera

# variavel responsavel pelas configurações do mediapipe
hand = mp.solutions.hands
Hand = hand.Hands(max_num_hands=1)  # essa variavel vai ser responsavel por fazer a detecção da mão dentro do video (1 porque eu so vou detectar uma mão)

mpDraw = mp.solutions.drawing_utils  # variavel que ira desenhar a ligação dos pontos na mão

while True:
    check, img = video.read()  # "lendo/abrindo" o video

    # por padrão, a imagem que nos recebemos da webcam é no formato BGR. mas nos precisamos converter ela para RGB
    # para que assim conseguimos processar essa imagem com o mediapipe
    imgRGB =  cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    # agora iremos processar a imagem com o mediapipe
    results =  Hand.process(imgRGB)

    # pegando as coordenadas dos pontos da mão
    handPoints = results.multi_hand_landmarks

    # percorrendo a varialvel handPoints e retornando as coordenadas para cada um dos pontos
    # mas antes, precisamos ter a certeza de que a variavel handPoints nao esta vazia
    
    # extraindo as dimensões da imagem
    h,w,_ = img.shape
    pontos = []
    if handPoints:
        for points in handPoints:
            # print(points)

            # agora que nos temos as coordenadas dos pontos, vamos desenhar eles dentro da imagem
            mpDraw.draw_landmarks(img, points, hand.HAND_CONNECTIONS)
            # draw_landmarks(a imagem em questão, os pontos, e o tipo de desenho que queremos ver)


        # enumerando cada ponto da mão
        for id, coord in enumerate(points.landmark):
            # o landmarks retorna uma coordenada dentro de uma proporção dentro da imagem
            # mas nos precisamos converter esses pontos em pixeis, para que assim nos possamos utilizar eles dentro da imagem
            # e para converter em pixeis, nos precisamos das dimensões da imagem que esta sendo gerada pela webcam
            
            cx = int(coord.x * w)
            cy = int(coord.y * h)

            cv2.putText(img, str(id), (cx, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            pontos.append((cx, cy))

        # lista que ira conter os pontos superiores de cada dedo
        dedos = [8, 12, 16, 20]

        # verificando se a variavel points nao esta vazia
        contador = 0
        if pontos: 

            # percorrendo a lista dos dedos
            for i in dedos:

                # um ponto esta no i e outro dois pontos a baixo
                if pontos[i][1] < pontos[i-2][1]:   # extraindo apenas a coordenada do y, por isso [1] ([0] eh do x)
                    contador += 1

            
            # fazendo a logica para o dedão
            if pontos[4][0] < pontos[2][0]:  # a logica eh sempre comparar com dois pontos a baixo do ponto extremo do dedo, por isso 4 e 2
                contador += 1 


        #print(contador)
        # inserindo essa informação dentro da imagem

        cv2.rectangle(img, (80, 10), (200,110), (255, 0, 0), -1)
        cv2.putText(img,str(contador),(100,100),cv2.FONT_HERSHEY_SIMPLEX,4,(255,255,255),5)

    else:

        print("Nenhuma mão na imagem foi processada!")

    # mostrando a imagem
    cv2.imshow("image", img)
    cv2.waitKey(1)  #definindo um delay de 1 segundo


