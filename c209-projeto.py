import cv2

arqCasc1 = r'CAMINHO_DO_ARQUIVO\haarcascade_frontalface_default.xml'
arqCasc2 = r'CAMINHO_DO_ARQUIVO\haarcascade_eye.xml'


faceCascade1 = cv2.CascadeClassifier(arqCasc1) #classificador para o rosto
faceCascade2 = cv2.CascadeClassifier(arqCasc2) #classificador para os olhos

# Source data
img_file = "img1.jpg"

# create an openCV image
imagem = cv2.imread(img_file)

faces = faceCascade1.detectMultiScale(
    imagem,
    minNeighbors=20,
    minSize=(30, 30),
maxSize=(300,300)
)

olhos = faceCascade2.detectMultiScale(
    imagem,
    minNeighbors=20,
    minSize=(10, 10),
maxSize=(90,90)
)

# Desenha um retangulo nas faces e olhos detectados
for (x, y, w, h) in faces:
    cv2.rectangle(imagem, (x, y), (x+w, y+h), (0, 255, 0), 4)

for (x, y, w, h) in olhos:
    cv2.rectangle(imagem, (x, y), (x+w, y+h), (255, 0, 0), 2)


# Finally display the image with the markings
cv2.imshow('my detection',imagem)

# wait for the keystroke to exit
cv2.waitKey()

