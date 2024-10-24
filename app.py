import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import AutoModelForImageClassification

# Load the pre-trained ViT model from Hugging Face
@st.cache_resource
def load_model():
    model = AutoModelForImageClassification.from_pretrained("nateraw/vit-base-patch16-224-cifar10")
    return model

# Function to preprocess the image for the model
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))  # Resize to model input size
    image = image / 255.0  # Normalize pixel values
    image = np.transpose(image, (2, 0, 1))  # Change to (channels, height, width)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return torch.tensor(image, dtype=torch.float32)

# Function to load the face detection model
def load_face_detector():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return face_cascade

# Function to detect faces in the image
def detect_faces(image, face_cascade):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return faces

# Function to process the image and make deep fake predictions
def process_image(image, faces, model):
    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        processed_face = preprocess_image(face)
        processed_face = processed_face.to(device)
        with torch.no_grad():
            outputs = model(processed_face)
        predictions = torch.softmax(outputs.logits, dim=1)
        label = torch.argmax(predictions, dim=1).item()
        confidence = predictions[0][label].item()
        label_name = 'Real' if label == 1 else 'Fake'
        st.write(f"Face: {label_name}, Confidence: {confidence:.2f}")
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return image

# Initialize Streamlit app
st.title("Deep Fake Detection")

# Load models
face_cascade = load_face_detector()
model = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# File uploader for the user to upload an image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    image = Image.open(uploaded_file)
    image_np = np.array(image.convert('RGB'))
    image_np_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # Detect faces in the image
    faces = detect_faces(image_np_bgr, face_cascade)
    
    # Process the image and draw face detection rectangles
    result_image = process_image(image_np_bgr, faces, model)
    
    # Convert back to RGB for displaying
    result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    
    # Display the processed image
    st.image(result_image_rgb, caption="Processed Image with Detections", use_column_width=True)
    
    # Display face detection information
    if len(faces) > 0:
        st.write("Faces detected: ", len(faces))
    else:
        st.write("No faces detected in the image.") 