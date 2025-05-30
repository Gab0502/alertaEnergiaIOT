import os
os.environ["QT_QPA_PLATFORM"] = "xcb"  # sempre antes do import cv2

import cv2

url = "rtsp://admin:Teste.Teste@192.168.15.18:554/h264/ch1/main/av_stream"

cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Não foi possível abrir o stream RTSP")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Falha ao ler frame")
        break

    cv2.imshow("Câmera IP (RTSP)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
