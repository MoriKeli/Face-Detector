import cv2 as cv

cascade_file = cv.CascadeClassifier('face.xml')

image_path = cv.imread("faces/trainer-2.jpg")
gray_image = cv.cvtColor(image_path, cv.COLOR_BGR2GRAY)

faces = cascade_file.detectMultiScale(gray_image, 1.1, 7)
print(f'{len(faces)} face(s) detected...')

while True:
    for (x, y, w, h) in faces:
        cv.rectangle(image_path, (x, y), (x+w, y+h), (0, 255, 0), 3)

    if image_path.shape[0] > 300 and image_path.shape[1] > 300:
        cv.imshow("Detected faces", cv.resize(image_path, (700, 500)))
    else:
        cv.imshow("Detected faces", image_path)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break
