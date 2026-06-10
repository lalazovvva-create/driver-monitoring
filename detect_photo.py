import cv2

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

image = cv2.imread("sample.jpg")

if image is None:
    print("Фото табылмады")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    roi_gray = gray[y:y+h, x:x+w]

    eyes = eye_cascade.detectMultiScale(roi_gray)

    if len(eyes) == 0:
        status = "EYES CLOSED"
        color = (0, 0, 255)
    else:
        status = "EYES OPEN"
        color = (0, 255, 0)

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(image,
                          (x+ex, y+ey),
                          (x+ex+ew, y+ey+eh),
                          (255, 0, 0), 2)

    cv2.putText(image, status, (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, color, 2)

print("Faces:", len(faces))

cv2.imshow("Photo Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()