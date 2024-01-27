# -------------------------------------------------------------------------------------------------------------------------------#
# Rastreamento de Objeto com OpenCV usando o OpticalFlow Dense

# O Optical Flow Denso (Dense Optical Flow) é uma técnica de visão computacional que calcula o movimento aparente dos pixels em uma 
# sequência de imagens. Diferente do Optical Flow tradicional, que opera em pontos específicos, o Dense Optical Flow calcula o vetor 
# de movimento para todos os pixels da imagem. Isso proporciona uma representação mais detalhada do movimento, sendo útil em aplicações 
# como rastreamento de objetos, análise de fluxo em vídeos e detecção de padrões de movimento em imagens.
# -------------------------------------------------------------------------------------------------------------------------------#

import cv2
import numpy as np

# Abre o vídeo de entrada
cap = cv2.VideoCapture("videos/walking.avi")

# Lê o primeiro quadro do vídeo e converte para escala de cinza
ret, first_frame = cap.read()
frame_gray_init = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# Inicializa a imagem HSV (Matiz, Saturação, Valor) para representar o fluxo óptico
hsv = np.zeros_like(first_frame)
hsv[..., 1] = 255  # Define a saturação para o máximo

# Loop principal
while True:
    # Lê um novo quadro do vídeo e converte para escala de cinza
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calcula o fluxo óptico usando o método Farneback
    flow = cv2.calcOpticalFlowFarneback(
                                        frame_gray_init, # primeiro Frame
                                        frame_gray,      # próximo frame
                                        None,            # Vetor do ponto 2D, caso tenha uma direção definida
                                        0.5,             # Escala da piramide de cores
                                        3,               # Quantidade de niveis da piramide
                                        15,              # Tamanho da janela
                                        3,               # iteração a cada nivel da pirâmide
                                        5,               # Tamanho da vizinhança do pixel para a expansão polinomial
                                        1.2,             # desvio padrão
                                        0                # flags
                                        )

    # Converte as coordenadas do fluxo óptico para magnitude e ângulo
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # hsv[...,0] = altera a matriz H (matiz)
    # hsv[...,1] = altera a matriz S (saturação)
    # hsv[...,2] = altera a matriz V (intensidade)

    # Converte o ângulo para a faixa de valores aceitável pelo espaço de cor HSV
    hsv[..., 0] = angle * (180 / (np.pi / 2))

    # Normaliza a magnitude para o intervalo de valores 0-255
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # Converte a imagem HSV de volta para o espaço de cor BGR
    final = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Exibe o resultado do fluxo óptico denso
    cv2.imshow('Dense optical flow', final)

    # Aguarda a tecla 'Enter' para encerrar o loop
    if cv2.waitKey(1) == 13:
        break

    # Atualiza o quadro de referência para o próximo loop
    frame_gray_init = frame_gray

# Libera os recursos e fecha a janela
cap.release()
cv2.destroyAllWindows()
