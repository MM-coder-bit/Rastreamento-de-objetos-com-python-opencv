# -------------------------------------------------------------------------------------------------------------------------------#
# Rastreamento de Objeto com OpenCV usando o CAMshift

# O algoritmo CAMshift (Continuous Adaptive Mean Shift) da OpenCV é uma extensão do algoritmo de rastreamento de objetos Mean Shift. 
# Ele é usado para rastrear objetos em sequências de vídeo. O CAMshift ajusta dinamicamente o tamanho e a forma da janela de busca 
# com base na distribuição de pixels dentro da janela. Isso permite que o algoritmo se adapte a mudanças na escala e na orientação 
# do objeto ao longo do tempo, tornando-o mais robusto para o rastreamento contínuo em vídeos
# -------------------------------------------------------------------------------------------------------------------------------#

# Importar bibliotecas necessárias
import numpy as np  # Para operações com arrays
import cv2  # Para processamento de imagens
from imutils.video import VideoStream  # Para acesso à webcam
import time  # Para pausas e temporizações

# 1. Iniciar a captura de vídeo
cap = VideoStream(src=0).start()  # Iniciar a transmissão da webcam
time.sleep(1.0)  # Pausa para a câmera iniciar

cap = cv2.VideoCapture(1)

# 2. Selecionar a região de interesse (ROI)
ret, frame = cap.read()  # Capturar um frame inicial
bbox = cv2.selectROI(frame, False)  # Permitir ao usuário selecionar a ROI
x, y, w, h = bbox  # Extrair coordenadas da ROI
track_window = (x, y, w, h)  # Criar uma tupla para armazenar a janela de rastreamento

# 3. Extrair a ROI e converter para o espaço de cores HSV
roi = frame[y:y+h, x:x+w]  # Recortar a ROI do frame

# Converter a ROI para o espaço de cores HSV (para melhor análise de cores)
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

# 4. Calcular o histograma da ROI
roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])  # Calcular o histograma do canal H
roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)  # Normalizar o histograma

# 5. Configurar critérios de parada para o rastreamento
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)  # Critérios de parada para o CamShift

# 6. Loop de rastreamento
while True:
    ret, frame = cap.read()  # Capturar um frame

    if ret == True:
        # Converter o frame para o espaço de cores HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Aplicar backprojection para encontrar regiões com cores semelhantes à ROI
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0,180], 1)

        # Aplicar o algoritmo CamShift para rastrear o objeto
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)

        # Obter os pontos da caixa delimitadora do objeto rastreado
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)  # Converter os pontos para inteiros

        # Desenhar um polígono ao redor do objeto rastreado
        img2 = cv2.polylines(frame, [pts], True, 255, 2)  # Linha branca, grossura 2

        # Exibir o frame com o rastreamento
        cv2.imshow('Camshift Rastreado', img2)

        # Sair do loop ao pressionar Enter
        if cv2.waitKey(1) == 13:
            break
    else:
        break

# 7. Finalizar a captura e liberar recursos
cv2.destroyAllWindows()
cap.release()
