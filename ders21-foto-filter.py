import cv2 as cv
import numpy as np
import gradio as gr

#Çeşitli Filtre fonksiyonları tanımla
def gaussian_blur(frame):
    return cv.GaussianBlur(frame, (15,15), 0)

def sharpen(frame):
    kernel=np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    return cv.filter2D(frame, -1, kernel)

def vintage(frame):
    kernel=np.array([[0.0722, 0.7152, 0.2126],
                     [0.393, 0.769, 0.189]])
    return cv.transform(frame, kernel)

def cartoon(frame):
    gray=cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray=cv.medianBlur(gray, 5)
    edges=cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 9, 9)
    color=cv.bilateralFilter(frame, 9, 300, 300)
    cartoon=cv.bitwise_and(color, color, mask=edges)
    return cartoon

#filtre uyuglama fonksiyon
def apply_filter(filter_name,input_image=None):
    if input_image is not None:
        frame=input_image
    else:
        cap=cv.VideoCapture(0)
        ret,frame=cap.read()
        cap.release()
        if not ret:
            return "Web cam den görüntü alınamıyor"

    if filter_name=="Gaussian Blur":
        return gaussian_blur(frame)
    elif filter_name=="Sharpen":
        return sharpen(frame)
    elif filter_name=="Vintage":
        return vintage(frame)
    elif filter_name=="Cartoon":
        return cartoon(frame)
    else:
        return frame
        
#Gradio Arayüz
with gr.Blocks() as demo:
    gr.Markdown("# OpenCV Foto Filtre")
    #filtre seçenekleri
    filter_name=gr.Dropdown(
        label="Filtre Seçin",
        choices=["Gaussian Blur", "Sharpen", "Vintage", "Cartoon"],
        value="Gaussian Blur"
    )
    interactive=True
    #giriş ve çıkış bileşenleri
        
    input_image=gr.Image(type="numpy", label="Resim Yükle")
    output_image=gr.Image(type="numpy", label="Filtreli Resim")
        
    #filtre uygula butonu 
    apply_button=gr.Button("Filtre Uygula")

    #Butona tıklayınca filtre uygula
    apply_button.click(
            fn=apply_filter,
            inputs=[filter_name, input_image],
            outputs=output_image
        )
        #Arayüzü başlat
demo.launch()
                    
            
        
    


    