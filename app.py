import streamlit as st
import torch
import clip
from PIL import Image
import numpy as np

# -----------------------------
# Load CLIP model
# -----------------------------
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

model, preprocess, device = load_model()

# -----------------------------
# App UI
# -----------------------------
st.title("🔍 DeepSearch AI")
st.write("Upload an image and search using text prompts using CLIP.")

# Upload image
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image using PIL
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    image_input = preprocess(image).unsqueeze(0).to(device)

    # Text input
    text_input = st.text_input("Enter search query", "a photo of a cat")

    if st.button("Search"):
        with st.spinner("Processing..."):

            # Tokenize text
            text_tokens = clip.tokenize([text_input]).to(device)

            # Encode image and text
            with torch.no_grad():
                image_features = model.encode_image(image_input)
                text_features = model.encode_text(text_tokens)

                # Normalize
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)

                # Similarity
                similarity = (image_features @ text_features.T).item()

            st.success(f"Similarity Score: {similarity:.4f}")

# -----------------------------
# Optional: Debug info
# -----------------------------
st.sidebar.write("Device:", device)