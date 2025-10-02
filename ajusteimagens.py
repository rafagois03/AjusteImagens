import os
import time
import tempfile
import numpy as np
import cv2
import streamlit as st
from PIL import Image, ImageEnhance
import zipfile
from io import BytesIO  # 👈 faltava importar


# Tenta importar PyMuPDF (fitz)
try:
    import fitz  # PyMuPDF
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False

# ===========================
# Funções utilitárias
# ===========================
def deskew_image(pil_img):
    try:
        img_np = np.array(pil_img)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) if img_np.ndim == 3 else img_np
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=80)
        if lines is None:
            return pil_img
        angles = []
        for line in lines:
            rho, theta = line[0]
            angle_deg = np.degrees(theta)
            if abs(angle_deg) < 30 or abs(angle_deg - 180) < 30:
                angles.append(angle_deg)
        if not angles:
            return pil_img
        avg_angle = np.mean(angles)
        if abs(avg_angle) < 0.5:
            return pil_img
        h, w = gray.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, -avg_angle, 1.0)
        cos, sin = abs(M[0, 0]), abs(M[0, 1])
        new_w, new_h = int(h * sin + w * cos), int(h * cos + w * sin)
        M[0, 2] += (new_w / 2 - center[0])
        M[1, 2] += (new_h / 2 - center[1])
        if img_np.ndim == 2:
            rotated = cv2.warpAffine(img_np, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        else:
            rotated = np.zeros((new_h, new_w, img_np.shape[2]), dtype=img_np.dtype)
            for i in range(img_np.shape[2]):
                rotated[:, :, i] = cv2.warpAffine(img_np[:, :, i], M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return Image.fromarray(rotated)
    except:
        return pil_img


def enhance_contrast_safely(img):
    try:
        if img.mode in ("L", "RGB"):
            return ImageEnhance.Contrast(img).enhance(2.0)
        elif img.mode == "RGBA":
            r, g, b, a = img.split()
            rgb = Image.merge("RGB", (r, g, b))
            rgb_enhanced = ImageEnhance.Contrast(rgb).enhance(2.0)
            r2, g2, b2 = rgb_enhanced.split()
            return Image.merge("RGBA", (r2, g2, b2, a))
        else:
            rgb = img.convert("RGB")
            return ImageEnhance.Contrast(rgb).enhance(2.0)
    except:
        return img


def process_image(file, output_folder):
    try:
        with Image.open(file) as img:
            img = img.copy()
            img = deskew_image(img)
            img = enhance_contrast_safely(img)
            w, h = img.size
            if w < 256 or h < 256:
                scale = max(256 / w, 256 / h)
                img = img.resize((int(w*scale), int(h*scale)), Image.Resampling.LANCZOS)
            save_path = os.path.join(output_folder, os.path.basename(file.name))
            img.save(save_path)
            return save_path, img
    except Exception as e:
        st.error(f"Erro ao processar imagem: {e}")
        return None, None


def convert_pdf_to_images(file, output_folder):
    if not FITZ_AVAILABLE:
        st.error("PyMuPDF não está disponível.")
        return []
    paths = []
    try:
        doc = fitz.open(stream=file.read(), filetype="pdf")
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
            temp_path = os.path.join(output_folder, f"pagina_{page_num+1}.jpg")
            pix.save(temp_path)
            paths.append(temp_path)
        return paths
    except Exception as e:
        st.error(f"Erro ao converter PDF: {e}")
        return []

# ===========================
# Interface Streamlit
# ===========================
st.title("📄 Processador de Imagens e PDFs - Projeto IA Canhotos GenMills")

uploaded_files = st.file_uploader(
    "Carregue imagens ou PDFs de canhotos que deseja processar com a IA",
    type=["jpg", "jpeg", "png", "pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    with tempfile.TemporaryDirectory() as tmpdir:
        processed_paths = []

        for file in uploaded_files:
            if file.name.lower().endswith(".pdf"):
                st.info(f"Convertendo PDF: {file.name}")
                imgs = convert_pdf_to_images(file, tmpdir)
                for img_path in imgs:
                    path, img = process_image(open(img_path, "rb"), tmpdir)
                    if img:
                        processed_paths.append(path)
            else:
                st.info(f"Processando imagem: {file.name}")
                path, img = process_image(file, tmpdir)
                if img:
                    processed_paths.append(path)

        # 👉 Compactar todos os arquivos processados em um único ZIP
        if processed_paths:
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zipf:
                for p in processed_paths:
                    zipf.write(p, arcname=os.path.basename(p))
            zip_buffer.seek(0)

            st.success(f"✅ {len(processed_paths)} arquivos processados com sucesso!")

            st.download_button(
                label="⬇️ Baixar todas as imagens (ZIP)",
                data=zip_buffer,
                file_name="imagens_processadas.zip",
                mime="application/zip"
            )
