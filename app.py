import streamlit as st
from PIL import Image
import numpy as np
import torch
from ultralytics import YOLO

# Load the YOLOv8 model
@st.cache_resource
def load_model():
    model = YOLO(r'C:\PSGiTech\imv project\ft_models\yolo_v83\weights\best.pt')  # Replace with your model file in yolov8 folder
    return model

model = load_model()

# Streamlit app layout
st.title("Kidney Stone Detection")
st.write("Upload an image for kidney stone detection.")

# Upload file
file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if file is not None:
    # Load and display the uploaded image
    image = Image.open(file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("Detecting...")

    # Convert the image to a format compatible with YOLOv8 (e.g., RGB format)
    image = np.array(image)

    # Perform detection
    results = model(image)

    # Display results
    annotated_image = results[0].plot()  # Draw detections on the image
    st.image(annotated_image, caption='Detection Results', use_column_width=True)

    # Map the detected class to a readable label
    class_mapping = {0: "Kidney stone detected", 1: "No kidney stone detected"}

    for box in results[0].boxes:
        class_id = int(box.cls.item())
        confidence = float(box.conf.item())
        label = class_mapping.get(class_id, "Unknown")
        st.write(f"Prediction: {label}, Confidence: {confidence:.2f}")