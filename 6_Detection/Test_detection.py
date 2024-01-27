# Importar a biblioteca OpenCV
import cv2

# Carregar a imagem a ser processada
image = cv2.imread('imagens/pessoas.jpg')

# Carregar o classificador em cascata para detecção de corpos inteiros
detector = cv2.CascadeClassifier('Detection/cascade/fullbody.xml')

# Converter a imagem para escala de cinza
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imshow("Pessoas", image_gray)  

# Detectar corpos inteiros na imagem usando o classificador em cascata
detections = detector.detectMultiScale(image_gray)

# Imprimir as coordenadas e dimensões das detecções
print(detections)
print(len(detections))

# Desenhar retângulos ao redor das detecções na imagem original
for (x, y, l, a) in detections:
    cv2.rectangle(image, (x, y), (x + l, y + a), (0, 255, 0), 2)

# Exibir a imagem com os retângulos das detecções
cv2.imshow("Detections", image)

# Aguardar até que uma tecla seja pressionada e então fechar a janela
cv2.waitKey(0)
cv2.destroyAllWindows()
