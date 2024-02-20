import cv2
import matplotlib.pyplot as plt

# Load image
imagePath = 'E:/Face Detection/Facedetection/Facedetection-master/test.jpg'
img = cv2.imread(imagePath)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

# Convert image to grayscale
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Provide path to the haarcascades directory
haarcascades_path = 'E:/Face Detection/Facedetection/Facedetection-master/'

# Load the face classifier
face_classifier = cv2.CascadeClassifier(haarcascades_path + "haarcascade_frontalface_default.xml")

# Detect faces
faces = face_classifier.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

# Draw rectangles around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (0, 255, 0), 4)

# Display the image
plt.figure(figsize=(20, 10))
plt.imshow(img_rgb)
plt.axis('off')
plt.show()
