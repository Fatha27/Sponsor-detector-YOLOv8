# Dataset

### Source:
The dataset consists of 1,425 images of Real Madrid and Barcelona players wearing jerseys, sourced from Roboflow.
### Classes: 
The target classes for detection are:
- Nike
- Adidas
- Spotify
- Emirates
  
## Preprocessing: 
Data augmentation techniques were applied to enhance the dataset, and the images were annotated using Roboflow's annotation tools.
## Model 
### Architecture: 
A custom YOLOv8 model was trained on the dataset to accurately detect the four target classes.
### Training: 
The model was trained to achieve optimal performance in recognizing the sponsor logos under various conditions.
### Results:
![image](https://github.com/user-attachments/assets/11b896d6-694f-4aa8-a34e-972909c476ae)
- The performance metrics can be improved by training for more epochs or on a larger dataset. 
## Backend
### API: 
A FastAPI backend was developed to handle image uploads and model inference. The backend processes the input images and returns the detection results.
![image](https://github.com/user-attachments/assets/56f0c89c-3b40-4705-8c64-86c228a2faa7)

## Frontend
### Interface: 
A Streamlit application provides a user-friendly interface for uploading images and displaying the detection results.
![image](https://github.com/user-attachments/assets/bf6ac2bd-b766-441e-949a-9f8b4cb652e6)
![image](https://github.com/user-attachments/assets/51208452-0cd3-47b2-a338-3920970eac7a)

### Features: 
Users can upload an image, and the detected sponsor logos will be highlighted with bounding boxes and class labels.
### View the Model on Roboflow
You can view the uploaded model on Roboflow for detailed inference, information and additional insights. Visit the following link to explore the dataset, annotations, and more: 
### [Roboflow Project Link](https://universe.roboflow.com/syed-fathaullah/sponsor-detector/model/2)  
