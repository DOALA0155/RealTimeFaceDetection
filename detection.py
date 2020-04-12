import cv2
import face_recognition

video_capture = cv2.VideoCapture(0)
eye_cascade = cv2.CascadeClassifier("../Cv2DetectionData/haarcascade_eye.xml")

face_locations = []

while True:
    ret, img = video_capture.read()
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rgb_img = img[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_img)

    for top, left, bottom, right in face_locations:
        cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)

        roi_color = img[top: bottom, right: left]

        roi_gray = gray[top: bottom, right: left]

        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
