import os
import time
import faiss
import torch
import streamlit as st
from PIL import Image
from torchvision import transforms
import clip
# Load the pre-trained CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
@st.cache_resource
def load_clip_model(device):
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess

model, preprocess = load_clip_model(device)
# Define a function to extract features from images using CLIP
@st.cache_resource
@st.cache_data
def extract_image_features(image_paths):
    image_features = []
    for image_path in image_paths:
        image = Image.open(image_path)
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_feature = model.encode_image(image_tensor)
            image_feature /= image_feature.norm(dim=-1, keepdim=True)
            image_features.append(image_feature)
    return torch.cat(image_features)

# Define a function to extract features from text using CLIP
@st.cache_resource
@st.cache_data
def extract_text_features(text):
    text_tensor = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tensor)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features

# Load the images from the MS-COCO 2017 Val dataset and extract their features
image_dir = "val2017"
image_paths = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir)]
image_features = extract_image_features(image_paths)

# Index the image features using FAISS for efficient similarity search
index = faiss.IndexFlatL2(image_features.shape[1])
index.add(image_features.cpu().numpy())

# Create a Streamlit app for the user interface
st.title("Multi-modal Search Engine with CLIP and FAISS")

query = st.text_input("Enter your query:")
uploaded_image = st.file_uploader("Upload an image:", type=["png", "jpg", "jpeg"])
text_search_button = st.button("Search with text")
image_search_button = st.button("Search with image")

if image_search_button and uploaded_image is not None:
    # Convert the uploaded image to a PIL Image
    image = Image.open(uploaded_image)
    start_time = time.time()
    # Extract features from the uploaded image using CLIP
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        query_features = model.encode_image(image_tensor)
    # Use FAISS to find the top-20 most similar images to the query

    distances, indices = index.search(query_features.cpu().numpy(), k=20)
    end_time = time.time()
    st.write(f"Processing time: {end_time - start_time} seconds")
    j = 0
    for i in range(5):
        cols = st.columns(4) # number of columns in each row! = 2
        for k in range(4):
            cols[k].image(Image.open(image_paths[indices[0][j]]), use_column_width=True)
            j += 1

if text_search_button and query is not None:
    # Extract features from the query using CLIP
    start_time = time.time()
    query_features = extract_text_features(query)
    distances, indices = index.search(query_features.cpu().numpy(), k=20)
    end_time = time.time()
    st.write(f"Processing time: {end_time - start_time} seconds")
    j = 0
    for i in range(5):
        cols = st.columns(4) # number of columns in each row! = 2
        for k in range(4):
            cols[k].image(Image.open(image_paths[indices[0][j]]), use_column_width=True)
            j += 1