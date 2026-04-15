import cv2 as cv
print (cv.__version__)

#kameradan görüntü al
cap=cv.VideoCapture(0) # 0 varsayılan kamera
while True:
    ret, frame=cap.read()
    #görüntüyü döndür
    cv.imshow("Org", frame) #ekrana yazdır
    frame_dikey=cv.flip(frame, 0) #1 yatay döndürür, 0 dikey döndürür -1 hem yatay hem dikey döndürür
    cv.imshow("Dikey", frame_dikey) #ekrana yazdır
    #q tusuna basılırsa döngüden çık
    if cv.waitKey(1) & 0xFF == ord("q"):
        break
cap.release() #kamerayı serbest bırak
cv.destroyAllWindows() #tüm pencereleri kapat
    

