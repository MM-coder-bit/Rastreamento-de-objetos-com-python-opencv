import cv2
import numpy as np

# Abre o vídeo de entrada
cap = cv2.VideoCapture("videos/walking.avi")

# Parâmetros para o método Shi-Tomasi usado para encontrar pontos de interesse iniciais
parameters_shitomasi = dict(maxCorners=100, qualityLevel=0.3, minDistance=7)

# Parâmetros para o método Lucas-Kanade usado para calcular o fluxo óptico
parameters_lucas_kanade = dict(winSize=(15, 15), maxLevel=2,
                               criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Gera cores aleatórias para visualização dos pontos
colors = np.random.randint(0, 255, (100, 3))

# Lê o primeiro quadro do vídeo
ret, frame = cap.read()

# Converte o primeiro quadro para escala de cinza
frame_gray_init = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Encontra pontos de interesse usando o método Shi-Tomasi
edges = cv2.goodFeaturesToTrack(frame_gray_init, mask=None, **parameters_shitomasi)

# Cria uma máscara para desenhar os vetores do fluxo óptico
mask = np.zeros_like(frame)

while True:
    # Lê um novo quadro do vídeo
    ret, frame = cap.read()

    # Converte o quadro para escala de cinza
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calcula o fluxo óptico entre os quadros usando o método Lucas-Kanade
    new_edges, status, errors = cv2.calcOpticalFlowPyrLK(frame_gray_init, frame_gray, edges, None,
                                                         **parameters_lucas_kanade)

    # Filtra os pontos de acordo com o status
    news = new_edges[status == 1]
    olds = edges[status == 1]

    # Desenha linhas e círculos nos quadros para visualizar o fluxo óptico
    for i, (new, old) in enumerate(zip(news, olds)):
        a, b = new.ravel()
        c, d = old.ravel()

        # Desenha uma linha indicando o vetor do fluxo óptico
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), colors[i].tolist(), 2)

        # Desenha um círculo nos pontos de interesse
        frame = cv2.circle(frame, (int(a), int(b)), 5, colors[i].tolist(), -1)

    # Adiciona a máscara ao quadro para visualização
    img = cv2.add(frame, mask)

    # Exibe o quadro resultante
    cv2.imshow('Optical flow', img)

    # Aguarda a tecla 'Enter' para encerrar o loop
    if cv2.waitKey(1) == 13:
        break

    # Atualiza o quadro de referência e os pontos de interesse
    frame_gray_init = frame_gray.copy()
    edges = news.reshape(-1, 1, 2)

# Fecha todas as janelas e libera os recursos
cv2.destroyAllWindows()
cap.release()
