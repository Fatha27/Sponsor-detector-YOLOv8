# app.py
import streamlit as st
import requests
from PIL import Image
import io

st.title("YOLOv8 Object Detection-Sponsor Detector")

# File upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Input fields
min_confidence = st.slider("Min Confidence", 0.0, 1.0, 0.4)
max_overlap = st.slider("Max Overlap", 0.0, 1.0, 0.3)

if uploaded_file is not None:
    if st.button("Detect Sponsors"):
        # Prepare the data for the API request
        files = {"file": ("image.jpg", uploaded_file.getvalue(), "image/jpeg")}
        data = {
            "min_confidence": str(min_confidence),
            "max_overlap": str(max_overlap)
        }
        
        # Make the API request
        response = requests.post("http://localhost:8000/detect/", files=files, data=data)
        
        if response.status_code == 200:
            # Display the result image
            result_image = Image.open(io.BytesIO(response.content))
            st.image(result_image, caption="Object Detection Result", use_column_width=True)
        else:
            st.error("Error in object detection. Please try again.")

st.markdown("---")
st.text("Powered by YOLOv8 and FastAPI")
