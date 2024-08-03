# main.py
from fastapi import FastAPI, File, HTTPException, UploadFile, Form
from fastapi.responses import FileResponse
import shutil
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict

app = FastAPI()

# Load your pretrained YOLOv8 model
model = YOLO('best.pt')

# Predefined classes
PREDEFINED_CLASSES = ["nike", "adidas", "emirates", "spotify"]

@app.post("/detect/")
async def detect_objects(
    file: UploadFile = File(...),
    min_confidence: float = Form(0.4),
    max_overlap: float = Form(0.3)
):
    try:
        # Save the uploaded file temporarily
        temp_image_path = "temp_image.jpg"
        with open(temp_image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Perform object detection
        results = model(temp_image_path, conf=min_confidence, iou=max_overlap)
        
        # Access the first result and its boxes
        boxes = results[0].boxes.xyxy
        classes = results[0].boxes.cls
        confidences = results[0].boxes.conf
        
        # Color map for different classes
        color_map = defaultdict(lambda: "white", {
            'nike': 'orange',
            'adidas': 'blue',
            'emirates': 'green',
            'spotify': 'purple',
        })
        
        # Open the image
        img = Image.open(temp_image_path)
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype("arial.ttf", 22)  
        
        # Draw bounding boxes and labels
        for box, cls, conf in zip(boxes, classes, confidences):
            class_name = results[0].names[int(cls)]
            if class_name.lower() not in PREDEFINED_CLASSES:
                continue
            
            label = f"{class_name} {conf:.2f}"
            color = color_map[class_name.lower()]
            
            # Draw rectangle
            draw.rectangle(box.tolist(), outline=color, width=3)
            
            # Calculate text size
            text_size = draw.textsize(label, font=font)
            
            # Draw text background
            text_bg_position = [box[0], box[1] - text_size[1], box[0] + text_size[0], box[1]]
            draw.rectangle(text_bg_position, fill=color)
            
            # Draw text
            draw.text((box[0], box[1] - text_size[1]), label, fill="white", font=font)
        
        # Save and return the result image
        result_image_path = "result.jpg"
        img.save(result_image_path)
        return FileResponse(result_image_path)
    except Exception as e:
        print(f"Error in object detection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in object detection: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
