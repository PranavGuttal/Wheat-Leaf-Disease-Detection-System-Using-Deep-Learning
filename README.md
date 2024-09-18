# Wheat-Leaf-Disease-Detection-System-Using-Deep-Learning

Project Overview
This project aims to develop an automated system for detecting diseases in wheat leaves using deep learning techniques. The system leverages the YOLOv8 object detection model to identify and classify wheat leaf diseases from images, assisting in timely diagnosis and improving crop management.

Features
Fast and Accurate Detection: Uses the YOLOv8 model for real-time detection of diseases in wheat leaves.
Image Processing Pipeline: Preprocessing of input images using the pathlib, PIL, and pandas libraries for improved accuracy.
User Interface: A simple and interactive user interface developed using Streamlit to allow users to upload wheat leaf images and view results.
Efficient Diagnosis: Provides quick identification of various wheat leaf diseases, assisting in better decision-making for farmers and agricultural experts.
Technologies Used
YOLOv8: For object detection and classification.
Python Libraries: pathlib, PIL, pandas, streamlit.
Deep Learning Framework: PyTorch.
Image Processing Techniques: Applied for image enhancement and preparation before feeding into the model.
Project Structure

├── data/                 # Dataset folder containing wheat leaf images
├── model/                # Trained YOLOv8 model files
├── src/                  # Source code files
│   ├── data_preprocessing.py    # Preprocesses input images
│   ├── model_training.py        # Code for training YOLOv8 model
│   ├── detection.py             # Code for detecting diseases using trained model
├── app.py                # Streamlit app for user interaction
├── README.md             # Project documentation
└── requirements.txt      # Required Python libraries

Installation
To run this project, clone the repository and install the necessary dependencies:

git clone https://github.com/yourusername/wheat-leaf-disease-detection.git
cd wheat-leaf-disease-detection
pip install -r requirements.txt

How to Use
Clone the repository and install the dependencies as mentioned above.
Run the Streamlit app to interact with the model and upload images of wheat leaves:

streamlit run app.py

Upload an image of a wheat leaf, and the system will display the detected disease along with its probability score.

Dataset
The dataset consists of labeled images of wheat leaves with various diseases. It was used to train the YOLOv8 model for disease classification. (Add dataset source if applicable).

Model Training
To retrain the model, ensure you have the dataset properly labeled and run the following:

python src/model_training.py


Future Enhancements
Add more wheat diseases to improve the model's generalization.
Implement a mobile version for on-field usage by farmers.
Improve detection accuracy by incorporating more advanced data augmentation techniques.
