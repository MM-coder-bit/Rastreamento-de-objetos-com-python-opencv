# -------------------------------------------------------------------------------------------------------------------------------#
# Rastreamento de Objeto com OpenCV usando o OpticalFlow Sparse

#O Optical Flow Sparse é uma técnica de visão computacional que calcula o movimento aparente de pontos-chave específicos em uma 
#sequência de imagens. Ao contrário do Optical Flow Denso, que calcula o vetor de movimento para todos os pixels da imagem, 
#o Optical Flow Sparse se concentra em pontos selecionados. Isso o torna mais eficiente computacionalmente, sendo adequado para 
#situações em que a densidade de pontos a serem rastreados pode ser reduzida sem comprometer a precisão do movimento estimado, 
#como em tarefas de rastreamento de objetos específicos em vídeos.
# -------------------------------------------------------------------------------------------------------------------------------#

import cv2
import numpy as np

# Inicializa a captura de vídeo da câmera (0 representa a câmera padrão)
cap = cv2.VideoCapture(0)

# Lê o primeiro quadro do vídeo e converte para escala de cinza
ret, frame = cap.read()
frame_gray_init = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Parâmetros para o método Lucas-Kanade usado para calcular o fluxo óptico
parameters_lucas_kanade = dict(winSize=(15, 15),
                               maxLevel=4,
                               criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Função de callback para seleção de ponto com o clique do mouse
def select_point(event, x, y, flags, params):
    global point, selected_point, old_points
    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x, y)
        selected_point = True
        old_points = np.array([[x, y]], dtype=np.float32)

# Cria uma janela para exibição do quadro e associa a função de callback
cv2.namedWindow('Frame')
cv2.setMouseCallback('Frame', select_point)

# Variáveis globais para o ponto selecionado, estado de seleção e pontos antigos
selected_point = False
point = ()
old_points = np.array([[]])

# Cria uma máscara inicializada com zeros
mask = np.zeros_like(frame)

while True:
    # Lê um novo quadro da câmera e converte para escala de cinza
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Se um ponto foi selecionado, realiza o rastreamento óptico
    if selected_point is True:
        # Desenha o ponto selecionado no quadro
        cv2.circle(frame, point, 5, (0, 0, 255), 2)

        # Calcula o fluxo óptico entre os quadros
        new_points, status, errors = cv2.calcOpticalFlowPyrLK(frame_gray_init,
                                                              frame_gray,
                                                              old_points,
                                                              None,
                                                              **parameters_lucas_kanade)
        # Atualiza o quadro de referência e os pontos antigos
        frame_gray_init = frame_gray.copy()
        old_points = new_points

        # Obtém as coordenadas dos pontos atuais e antigos
        x, y = new_points.ravel()
        j, k = old_points.ravel()

        # Desenha uma linha indicando o vetor do fluxo óptico
        mask = cv2.line(mask, (int(x), int(y)), (int(j), int(k)), (0, 255, 255), 2)
        
        # Desenha um círculo nos pontos de interesse
        frame = cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

    # Combina o quadro original com a máscara para visualização
    img = cv2.add(frame, mask)

    # Exibe o quadro resultante e a máscara
    cv2.imshow("Frame", img) # imagem concatenada
    cv2.imshow("Frame 2", mask) # Fundo escuro
    cv2.imshow("Frame 3", frame) # imagem Original

    # Aguarda a tecla 'Esc' para encerrar o loop
    key = cv2.waitKey(1)
    if key == 27:
        break

# Libera os recursos e fecha as janelas
cap.release()
cv2.destroyAllWindows()
