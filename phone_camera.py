import cv2

url="http://192.168.118.205:8080/video"

cap = cv2.VideoCapture(url)

while True:
    ret, frame = cap.read()

    if not ret:
        print("could not recieve frame")
        break

    cv2.imshow("phone Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()