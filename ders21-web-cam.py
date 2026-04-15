import cv2 as cv
print (cv.__version__)

#kameradan görüntü al
cap=cv.VideoCapture(0) # 0 varsayılan kamera
while True:
    ret, frame=cap.read()
    #gray=cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow("webcam", frame)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv.destroyAllWindows()
    

