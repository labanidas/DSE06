import cv2

# Load the Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the two images to compare
image1 = cv2.imread('images/pawan.jpg')
image2 = cv2.imread('images/flower.jpg')

# Convert both images to grayscale
gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Detect faces in both images
faces1 = face_cascade.detectMultiScale(gray_image1, scaleFactor=1.1, minNeighbors=5)
faces2 = face_cascade.detectMultiScale(gray_image2, scaleFactor=1.1, minNeighbors=5)

# If either of the images doesn't contain any faces, exit
if len(faces1) == 0 or len(faces2) == 0:
    print('Could not detect any faces in one or both images')
    exit()

# Get the first detected face from each image
(x1, y1, w1, h1) = faces1[0]
(x2, y2, w2, h2) = faces2[0]

# Extract the face region from both images
face1 = gray_image1[y1:y1+h1, x1:x1+w1]
face2 = gray_image2[y2:y2+h2, x2:x2+w2]

# Resize both face images to the same size
face1 = cv2.resize(face1, (100, 100))
face2 = cv2.resize(face2, (100, 100))

# Calculate the mean squared error (MSE) between the two face images
mse = ((face1 - face2) ** 2).mean()
print(mse)
# Define a threshold for determining if the faces match or not
# threshold = 2000
threshold = 100

# Compare the MSE with the threshold and print the result
if mse < threshold:
    print('The two faces match')
else:
    print('The two faces do not match')
