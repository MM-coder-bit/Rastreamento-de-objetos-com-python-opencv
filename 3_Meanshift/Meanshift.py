# pip install imutils
# -------------------------------------------------------------------------------------------------------------------------------#
# Rastreamento de Objeto com OpenCV usando o Meanshift

#O algoritmo Meanshift é uma técnica de segmentação de objetos em imagens. Ele ajusta iterativamente a posição de uma janela para 
#a região de maior densidade de pontos de uma distribuição de cores, buscando convergir para o centro (média) da região. 
#Isso é útil para rastreamento de objetos em movimento ou segmentação de regiões de interesse em uma imagem
# -------------------------------------------------------------------------------------------------------------------------------#

# Rastreamento de Objeto com OpenCV

# Importar bibliotecas necessárias
import cv2
import time
from imutils.video import VideoStream  # Para acesso à webcam

# 1. Iniciar a captura de vídeo
cap = VideoStream(src=0).start()  # Iniciar a transmissão da webcam
time.sleep(1.0)  # Pausa para a câmera iniciar

cap = cv2.VideoCapture(0)

# 2. Selecionar a região de interesse (ROI)
ret, frame = cap.read()  # Capturar um frame inicial
bbox = cv2.selectROI(frame, False)  # Permitir ao usuário selecionar a ROI
x, y, w, h = bbox  # Extrair coordenadas da ROI
track_window = (x, y, w, h)  # Criar uma tupla para armazenar a janela de rastreamento
print(track_window)  # Imprimir as coordenadas para verificação

# 3. Extrair a ROI e converter para o espaço de cores HSV
roi = frame[y:y+h, x:x+w]  # Recortar a ROI do frame
#cv2.imshow('ROI', roi)  # Exibir a ROI

# Converter a ROI para o espaço de cores HSV (para melhor análise de cores)
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
#cv2.imshow("ROI HSV", hsv_roi)  # Exibir a ROI no espaço HSV

# 4. Calcular o histograma da ROI
roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])  # Calcular o histograma do canal H

# Exibir o histograma da ROI (apenas para visualização)
import matplotlib.pyplot as plt
plt.hist(roi.ravel(), 180, [0, 180])  # Plotar o histograma
#plt.show()

# Normalizar o histograma
roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)  # Normalizar os valores de 0 a 255

# 5. Configurar critérios de parada para o rastreamento
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)  # Critérios de parada para o meanShift

# 6. Loop de rastreamento
while True:
    ret, frame = cap.read()  # Capturar um frame

    if ret == True:
        # Converter o frame para o espaço de cores HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Aplicar backprojection para encontrar regiões com cores semelhantes à ROI
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0,180], 1)

        # Aplicar o algoritmo meanShift para rastrear o objeto
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        # Atualizar as coordenadas da janela de rastreamento
        x, y, w, h = track_window

        # Desenhar um retângulo ao redor do objeto rastreado
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)  # Retangulo verde

        # Exibir os frames resultantes
        cv2.imshow('Meanshift tracking', frame)  # Frame com rastreamento
        cv2.imshow('dst', dst)  # Backprojection
        #cv2.imshow('ROI', roi)  # ROI original

        # Sair do loop ao pressionar Enter
        if cv2.waitKey(1) == 13:
            break
    else:
        break

# 7. Finalizar a captura e liberar recursos
cv2.destroyAllWindows()
cap.release()
