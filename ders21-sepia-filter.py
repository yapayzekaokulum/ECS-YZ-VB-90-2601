import cv2 as cv
import numpy as  np
cap=cv.VideoCapture(0)

def retro_filter(frame):
    frame_sepia=cv.transform(frame, np.matrix(
        [[0.272, 0.534, 0.131],
        [0.349, 0.686, 0.168],
        [0.393, 0.769, 0.189]]  
    ))

    #Görüntüyü normalize et
    frame_sepia=np.clip(frame_sepia, 0,255).astype(np.uint8)
    return frame_sepia

while True:
    ret,frame=cap.read()
    frame_sepia=retro_filter(frame)
    cv.imshow('Sepia',frame_sepia)
    if cv.waitKey(1) & 0xFF==ord('q'):
        break
cap.release()
cv.destroyAllWindows()