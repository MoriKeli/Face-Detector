import cv2 as cv
cascade_file = cv.CascadeClassifier('face.xml')

# webcam config
webcam = cv.VideoCapture(0)
webcam.set(3, 640)  # width
webcam.set(4, 480)  # height
webcam.set(10, 150)    # brightness

while True:
    is_successful, video = webcam.read()
    faces = cascade_file.detectMultiScale(video, 1.5, 1)
    for (x, y, w, h) in faces:
        cv.rectangle(video, (x, y), (x+w, y+h), (255, 0, 0), 3)
        print(f'{len(faces)} face(s) detected...')
        cv.imshow("Face detector", video)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv.destroyAllWindows()
