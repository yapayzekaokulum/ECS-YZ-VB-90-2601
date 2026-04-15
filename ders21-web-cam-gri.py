import cv2 as cv
print (cv.__version__)

#kameradan görüntü al
cap=cv.VideoCapture(0) # 0 varsayılan kamera
while True:
    ret, frame=cap.read()
    #gri formatta oku
    gray=cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow("webcam", gray)
    #q tusuna basılırsa döngüden çık
    if cv.waitKey(1) & 0xFF == ord("q"):
        break
cap.release() #kamerayı serbest bırak
cv.destroyAllWindows() #tüm pencereleri kapat
    

