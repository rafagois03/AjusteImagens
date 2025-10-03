import os
import time
import tempfile
import numpy as np
import cv2
import streamlit as st
from PIL import Image
import zipfile
from io import BytesIO

# Tenta importar PyMuPDF (fitz)
try:
    import fitz  # PyMuPDF
    FITZ_AVAILABLE = True
    st.success("✅ PyMuPDF (fitz) carregado com sucesso.")
except ImportError:
    FITZ_AVAILABLE = False
    st.warning("❌ PyMuPDF (fitz) NÃO encontrado. Conversão de PDF desabilitada.")


# ===========================
# Funções utilitárias
# ===========================
def deskew_image(pil_img):
    """Deskew otimizado para documentos com bordas brancas (PDFs)."""
    try:
        img_np = np.array(pil_img)
        if img_np.ndim == 2:
            gray = img_np
        else:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # --- Aumenta contraste local para destacar bordas ---
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # --- Detecta bordas ---
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # --- Detecta linhas ---
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=80)
        if lines is None:
            st.warning("⚠️ Nenhuma linha detectada para deskew. Mantendo original.")
            return pil_img

        angles = []
        for line in lines:
            rho, theta = line[0]
            angle_deg = np.degrees(theta)
            if abs(angle_deg) < 30 or abs(angle_deg - 180) < 30:
                angles.append(angle_deg)

        if len(angles) == 0:
            st.warning("⚠️ Nenhuma linha horizontal encontrada. Mantendo original.")
            return pil_img

        avg_angle = np.mean(angles)
        if abs(avg_angle) < 0.5:
            return pil_img

        st.info(f"📏 Ângulo detectado: {avg_angle:.2f}°")

        # --- Rotação com padding ---
        h, w = gray.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, -avg_angle, 1.0)

        cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        M[0, 2] += (new_w / 2 - center[0])
        M[1, 2] += (new_h / 2 - center[1])

        if img_np.ndim == 2:
            rotated = cv2.warpAffine(img_np, M, (new_w, new_h),
                                     flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        else:
            rotated = np.zeros((new_h, new_w, img_np.shape[2]), dtype=img_np.dtype)
            for i in range(img_np.shape[2]):
                rotated[:, :, i] = cv2.warpAffine(img_np[:, :, i], M, (new_w, new_h),
                                                  flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        return Image.fromarray(rotated)

    except Exception as e:
        st.error(f"⚠️ Erro no deskew: {e}. Retornando original.")
        return pil_img


def process_image(file, output_folder):
    """Processa uma imagem: deskew, redimensiona e salva."""
    try:
        with Image.open(file) as img:
            img = img.copy()
            st.info(f"🖼️ Processando imagem: {file.name}")

            # Deskew
            img = deskew_image(img)

            # Redimensiona se muito pequeno
            w, h = img.size
            if w < 256 or h < 256:
                scale = max(256 / w, 256 / h)
                new_w = max(int(w * scale), 256)
                new_h = max(int(h * scale), 256)
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                st.info(f"✅ Redimensionado para: {new_w}x{new_h}")

            save_path = os.path.join(output_folder, os.path.basename(file.name))
            img.save(save_path)

            return save_path, img
    except Exception as e:
        st.error(f"❌ Falha ao processar {file.name}: {e}")
        return None, None


def convert_pdf_to_images(file, output_folder):
    """Converte PDF em imagens com PyMuPDF (fitz)."""
    if not FITZ_AVAILABLE:
        st.error("🛑 Conversão de PDF desabilitada — PyMuPDF não disponível.")
        return []

    paths = []
    try:
        doc = fitz.open(stream=file.read(), filetype="pdf")
        total_pages = len(doc)
        st.info(f"✅ PDF aberto ({total_pages} páginas)")

        base_name = os.path.splitext(file.name)[0]

        for page_num in range(total_pages):
            page = doc.load_page(page_num)
            mat = fitz.Matrix(3, 3)  # ~300 DPI
            pix = page.get_pixmap(matrix=mat)

            temp_name = f"{base_name}_pagina_{page_num+1}_temp.jpg"
            temp_path = os.path.join(output_folder, temp_name)
            pix.save(temp_path)

            path, img = process_image(open(temp_path, "rb"), output_folder)
