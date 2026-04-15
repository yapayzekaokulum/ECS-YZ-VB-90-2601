import cv2
import numpy as np
import gradio as gr


def _to_bgr(img):
    if img is None:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def _clip(img):
    return np.clip(img, 0, 255).astype(np.uint8)


def f_original(img, **_):
    return img


def f_gaussian_blur(img, blur_ksize=15, **_):
    k = max(1, int(blur_ksize))
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(img, (k, k), 0)


def f_sharpen(img, **_):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)


def f_edge(img, edge_low=100, edge_high=200, **_):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, int(edge_low), int(edge_high))
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


def f_invert(img, **_):
    return cv2.bitwise_not(img)


def f_brightness(img, alpha=1.2, beta=30, **_):
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def f_grayscale(img, **_):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def f_sepia(img, **_):
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    return _clip(cv2.transform(img.astype(np.float32), kernel))


def f_fall(img, **_):
    kernel = np.array([[0.393, 0.769, 0.189],
                       [0.349, 0.686, 0.168],
                       [0.272, 0.534, 0.131]])
    return _clip(cv2.transform(img.astype(np.float32), kernel))


def f_cartoon(img, **_):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(img, 9, 300, 300)
    return cv2.bitwise_and(color, color, mask=edges)


def f_pencil_sketch(img, **_):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (21, 21), 0)
    sketch = cv2.divide(gray, 255 - blur, scale=256)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)


def f_oil_painting(img, **_):
    try:
        return cv2.xphoto.oilPainting(img, 7, 1)
    except Exception:
        return cv2.stylization(img, sigma_s=60, sigma_r=0.45)


def f_emboss(img, **_):
    kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
    embossed = cv2.filter2D(img, -1, kernel) + 128
    return _clip(embossed)


def f_cool(img, **_):
    b, g, r = cv2.split(img.astype(np.float32))
    b = np.minimum(b * 1.3, 255)
    r = r * 0.85
    return _clip(cv2.merge([b, g, r]))


def f_warm(img, **_):
    b, g, r = cv2.split(img.astype(np.float32))
    r = np.minimum(r * 1.3, 255)
    b = b * 0.85
    return _clip(cv2.merge([b, g, r]))


def f_vignette(img, vignette_strength=1.0, **_):
    rows, cols = img.shape[:2]
    kx = cv2.getGaussianKernel(cols, cols / (2 * max(0.5, vignette_strength)))
    ky = cv2.getGaussianKernel(rows, rows / (2 * max(0.5, vignette_strength)))
    mask = ky @ kx.T
    mask = mask / mask.max()
    out = img.astype(np.float32) * mask[..., None]
    return _clip(out)


def f_pixelate(img, pixel_size=12, **_):
    h, w = img.shape[:2]
    p = max(2, int(pixel_size))
    small = cv2.resize(img, (max(1, w // p), max(1, h // p)),
                       interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def f_hdr(img, **_):
    return cv2.detailEnhance(img, sigma_s=12, sigma_r=0.15)


def f_cyberpunk(img, **_):
    b, g, r = cv2.split(img.astype(np.float32))
    b = np.minimum(b * 1.4 + 20, 255)
    r = np.minimum(r * 1.2 + 15, 255)
    g = g * 0.7
    merged = _clip(cv2.merge([b, g, r]))
    return cv2.convertScaleAbs(merged, alpha=1.2, beta=5)


def f_vintage(img, **_):
    sepia = f_sepia(img)
    vign = f_vignette(sepia, vignette_strength=1.2)
    noise = np.random.randint(-15, 15, vign.shape, dtype=np.int16)
    return _clip(vign.astype(np.int16) + noise)


def f_mirror(img, **_):
    return cv2.flip(img, 1)


def f_kaleidoscope(img, **_):
    h, w = img.shape[:2]
    half = img[:, :w // 2]
    mirrored = cv2.flip(half, 1)
    top = np.hstack([half, mirrored])
    bottom = cv2.flip(top, 0)
    combined = np.vstack([top[:h // 2], bottom[h // 2:]])
    return cv2.resize(combined, (w, h))


def f_thermal(img, **_):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(gray, cv2.COLORMAP_JET)


def f_negative_film(img, **_):
    inv = cv2.bitwise_not(img)
    b, g, r = cv2.split(inv.astype(np.float32))
    r = np.minimum(r * 1.2, 255)
    return _clip(cv2.merge([b, g, r]))


FILTERS = {
    "Orijinal": f_original,
    "Gaussian Bulanıklık": f_gaussian_blur,
    "Keskinleştir": f_sharpen,
    "Kenar Tespiti": f_edge,
    "Negatif": f_invert,
    "Parlaklık/Kontrast": f_brightness,
    "Gri Tonlama": f_grayscale,
    "Sepya": f_sepia,
    "Sonbahar": f_fall,
    "Çizgi Film": f_cartoon,
    "Kurşun Kalem": f_pencil_sketch,
    "Yağlı Boya": f_oil_painting,
    "Kabartma": f_emboss,
    "Soğuk Ton": f_cool,
    "Sıcak Ton": f_warm,
    "Vinyet": f_vignette,
    "Piksel": f_pixelate,
    "HDR": f_hdr,
    "Cyberpunk": f_cyberpunk,
    "Vintage": f_vintage,
    "Ayna": f_mirror,
    "Kaleydoskop": f_kaleidoscope,
    "Termal": f_thermal,
    "Negatif Film": f_negative_film,
}


def apply_filter(image, filter_name, alpha, beta, blur_ksize,
                 edge_low, edge_high, vignette_strength, pixel_size):
    if image is None:
        return None
    img = _to_bgr(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    fn = FILTERS.get(filter_name, f_original)
    result = fn(
        img,
        alpha=alpha,
        beta=beta,
        blur_ksize=blur_ksize,
        edge_low=edge_low,
        edge_high=edge_high,
        vignette_strength=vignette_strength,
        pixel_size=pixel_size,
    )
    result = _to_bgr(result)
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)


with gr.Blocks(theme=gr.themes.Soft(), title="Gelişmiş Görüntü Filtreleri") as demo:
    gr.Markdown(
        """
        # 🎨 Gelişmiş Görüntü Filtreleri
        Web kameranı kullan veya bir görsel yükle — 24 farklı filtre ile anında deneyimle.
        Slider'lar bazı filtrelerin davranışını kontrol eder (Parlaklık, Bulanıklık, Kenar, Vinyet, Piksel).
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                label="Giriş (Kamera / Yükle)",
                sources=["upload", "webcam"],
                type="numpy",
            )
            filter_type = gr.Dropdown(
                label="Filtre",
                choices=list(FILTERS.keys()),
                value="Orijinal",
            )
            with gr.Accordion("⚙️ Parametreler", open=False):
                alpha = gr.Slider(0.1, 3.0, value=1.2, step=0.1, label="Kontrast (alpha)")
                beta = gr.Slider(-100, 100, value=30, step=1, label="Parlaklık (beta)")
                blur_ksize = gr.Slider(1, 51, value=15, step=2, label="Bulanıklık Çekirdek")
                edge_low = gr.Slider(0, 255, value=100, step=1, label="Kenar Alt Eşik")
                edge_high = gr.Slider(0, 255, value=200, step=1, label="Kenar Üst Eşik")
                vignette_strength = gr.Slider(0.3, 3.0, value=1.0, step=0.1, label="Vinyet Şiddeti")
                pixel_size = gr.Slider(2, 50, value=12, step=1, label="Piksel Boyutu")

        with gr.Column(scale=1):
            output_image = gr.Image(label="Sonuç", type="numpy")

    inputs = [input_image, filter_type, alpha, beta, blur_ksize,
              edge_low, edge_high, vignette_strength, pixel_size]

    for ctrl in inputs:
        ctrl.change(fn=apply_filter, inputs=inputs, outputs=output_image)

    gr.Markdown("### 💡 İpucu: Oil Painting için `opencv-contrib-python` paketi önerilir.")


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
