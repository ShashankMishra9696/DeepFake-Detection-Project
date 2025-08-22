import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from facenet_pytorch import MTCNN
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Import model loading function
from model import load_model

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = load_model()
model.to(DEVICE)
model.eval()

# Initialize MTCNN (exact same as in notebook)
mtcnn = MTCNN(
    select_largest=False,
    post_process=False,
    device=DEVICE
).eval()

# Streamlit UI
st.set_page_config(page_title="Deepfake Detections", layout="centered")
st.title("🧠 Deepfake Detection")
st.markdown("Upload a **face image**, and this app will predict if it's **Real** or **Fake** using an AI model.")

uploaded_file = st.file_uploader("📤 Upload a face image", type=["jpg", "jpeg", "png"])

# Predict only on button click
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    if st.button("🔍 Predict"):
        with st.spinner("Analyzing..."):
            face = mtcnn(image)
            if face is None:
                st.error("❌ No face detected!")
            else:
                # Preprocess
                face = face.unsqueeze(0)
                face = F.interpolate(face, size=(256, 256), mode='bilinear', align_corners=False)

                original_face = face.squeeze(0).permute(1, 2, 0).cpu().detach().int().numpy().astype('uint8')
                input_face = face.to(DEVICE).to(torch.float32) / 255.0
                face_img_for_cam = input_face.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()

                # GradCAM setup
                target_layers = [model.block8.branch1[-1]]
                targets = [ClassifierOutputTarget(0)]

                with GradCAM(model=model, target_layers=target_layers) as cam:
                    grayscale_cam = cam(input_tensor=input_face, targets=targets, eigen_smooth=True)[0]

                cam_overlay = show_cam_on_image(face_img_for_cam, grayscale_cam, use_rgb=True)
                combined_image = cv2.addWeighted(original_face, 1, cam_overlay, 0.5, 0)

                # Predict
                with torch.no_grad():
                    output = torch.sigmoid(model(input_face).squeeze(0))
                    confidence = output.item()
                    prediction = "🟢 REAL" if confidence < 0.5 else "🔴 FAKE"

                # Debug output (optional)
                st.write(f"Model raw output (sigmoid): `{confidence*100:.4f}`")

                # Results
                st.success(f"### Prediction: {prediction}")
                st.markdown(f"**Confidence:** `{confidence*100:.4f}%`")
                st.markdown(f"**Real:** `{100 - confidence*100:.4f}`")
                st.markdown(f"**Fake:** `{confidence*100:.4f}`")
                st.image(combined_image, caption="Grad-CAM Overlay", use_column_width=True)
