import cv2

capture = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('../Cv2DetectionData/haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('../Cv2DetectionData/haarcascade_smile.xml')

while True:
    ret, img = capture.read()
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        cv2.circle(img, (int(x + w / 2), int(y + h / 2)), int( w / 2), (255, 0, 0), 2)

        roi_gray = gray[y: y + h, x: x + w]
        smiles= smile_cascade.detectMultiScale(roi_gray, scaleFactor= 1.2, minNeighbors=10, minSize=(20, 20))

        if len(smiles) >0 :
            for(sx,sy,sw,sh) in smiles:
                cv2.circle(img, (int(x + sx + sw / 2), int(y + sy + sh / 2)), int(sw / 2), (0, 0, 255), 2)

    cv2.imshow("Video", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
