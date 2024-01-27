# Importar as bibliotecas OpenCV e sys, além de randint do módulo random
import cv2
import sys
from random import randint

# Criar um objeto de rastreamento CSRT (Discriminative Correlation Filter with Channel and Spatial Reliability)
tracker = cv2.TrackerCSRT_create()

# Abrir um vídeo para rastreamento
video = cv2.VideoCapture("videos/walking.avi")

# Verificar se o vídeo foi aberto corretamente
if not video.isOpened():
    print("Não foi possível abrir o vídeo")
    sys.exit()

# Ler o primeiro frame do vídeo
ok, frame = video.read()
if not ok:
    print('Não é possível ler o arquivo de vídeo')
    sys.exit()

# Carregar o classificador em cascata para detecção de corpos inteiros
cascade = cv2.CascadeClassifier('Detection/cascade/fullbody.xml')

# Função para detectar corpos inteiros usando o classificador em cascata
def detectar():
    while True:
        # Ler um novo frame do vídeo
        ok, frame = video.read()
        # Converter o frame para escala de cinza
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detectar corpos inteiros na imagem usando o classificador em cascata
        detection = cascade.detectMultiScale(frame_gray)
        # Desenhar retângulos ao redor das detecções na imagem
        for (x, y, l, a) in detection:
            cv2.rectangle(frame, (x, y), (x + l, y + a), (0, 0, 255), 2)
            cv2.imshow("Deteção", frame)
            # Verificar se a detecção foi realizada pelo classificador em cascata
            if x > 0:
                print('Detecção efetuada pelo haarcascade')
                return x, y, l, a

# Chamar a função de detecção e obter as coordenadas da detecção
bbox = detectar()

# Inicializar o rastreador com as coordenadas da detecção
ok = tracker.init(frame, bbox)
# Gerar uma cor aleatória para desenhar o retângulo de rastreamento
colors = (randint(0, 255), randint(0, 255), randint(0, 255))

# Loop principal de rastreamento
while True:
    # Ler um novo frame do vídeo
    ok, frame = video.read()
    if not ok:
        break

    # Atualizar o rastreador com o novo frame
    ok, bbox = tracker.update(frame)

    # Verificar se o rastreamento foi bem-sucedido
    if ok:
        # Converter os valores de ponto flutuante em inteiros e extrair as coordenadas
        (x, y, w, h) = [int(v) for v in bbox]
        # Desenhar um retângulo ao redor do objeto rastreado
        cv2.rectangle(frame, (x, y), (x + w, y + h), colors, 2, 1)
    else:
        # Em caso de falha no rastreamento, realizar detecção usando o classificador em cascata
        print('Falha no rastreamento. Será executado o detector haarcascade')
        bbox = detectar()
        # Inicializar um novo rastreador (MOSSE neste caso) com as novas coordenadas
        tracker = cv2.legacy.TrackerMOSSE_create()
        tracker.init(frame, bbox)

    # Exibir o frame com o retângulo de rastreamento
    cv2.imshow("Tracking", frame)

    # Aguardar até que uma tecla seja pressionada (27 corresponde à tecla 'ESC') e encerrar o loop se necessário
    k = cv2.waitKey(1) & 0XFF
    if k == 27:
        break