# Python In-built packages
from pathlib import Path
import PIL
import pandas as pd

# External packages
import streamlit as st

# Local Modules
import settings
import helper

# Setting page layout
st.set_page_config(
    page_title="Wheat Leaf Disease Detection using YOLOv8",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Main page heading
st.title("Wheat Leaf Disease Detection using YOLOv8")

# Sidebar
st.sidebar.header("ML Model Config")

# Model Options
model_type = st.sidebar.radio(
    "Select Task", ['Detection', 'Segmentation'])

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 40)) / 100

# Selecting Detection Or Segmentation
if model_type == 'Detection':
    model_path = Path(settings.DETECTION_MODEL)
elif model_type == 'Segmentation':
    model_path = Path(settings.SEGMENTATION_MODEL)

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)

##################################################################################
##################################################################

source_img = None
# If image is selected
if source_radio == settings.IMAGE:

    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))
    
    st.sidebar.header("Disease Treatment")

    # Selecting disease name
    selected_disease = st.sidebar.selectbox(
        "Select Disease",
        ["Wheat powdery mildew", "Wheat septoria", "Wheat stem rust", "Wheat yellow rust"]
    )


    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Default Image",
                         use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image",
                         use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Image',
                     use_column_width=True)
        else:
            if st.sidebar.button('Detect Disease'):
                res = model.predict(uploaded_image, conf=confidence)
                boxes = res[0].boxes 
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image',
                         use_column_width=True)
                try:
                    with st.expander("Detection Results"):
                        for box in boxes:
                            st.write(box.data)
                        ##########################
                        #########################
                except Exception as ex:
                    # st.write(ex)
                    st.write("No image is uploaded yet!")

    # Disease Treatment Section
            if st.sidebar.button("Treatment"):
                if selected_disease:
                    res = model.predict(uploaded_image, conf=confidence)
                    st.header(f"Treatment Recommendations for {selected_disease}")
                    treatment_recommendations = {
                        "Wheat powdery mildew": [
                            "Treatment recommendation 1 for wheat powdery mildew",
                            "Treatment recommendation 2 for wheat powdery mildew",
                            "Treatment recommendation 3 for wheat powdery mildew"
                        ],
                        "Wheat septoria": [
                            "Treatment recommendation 1 for wheat septoria",
                            "Treatment recommendation 2 for wheat septoria",
                            "Treatment recommendation 3 for wheat septoria"
                        ],
                        "Wheat stem rust": [
                            "Treatment recommendation 1 for wheat stem rust",
                            "Treatment recommendation 2 for wheat stem rust",
                            "Treatment recommendation 3 for wheat stem rust"
                        ],
                        "Wheat yellow rust": [
                            "Treatment recommendation 1 for wheat yellow rust",
                            "Treatment recommendation 2 for wheat yellow rust",
                            "Treatment recommendation 3 for wheat yellow rust"
                        ]
                    }
                    for i, recommendation in enumerate(treatment_recommendations[selected_disease], 1):
                        st.write(f"Treatment recommendation {i}: {recommendation}")

elif source_radio == settings.VIDEO:
    helper.play_stored_video(confidence, model)

elif source_radio == settings.WEBCAM:
    helper.play_webcam(confidence, model)

elif source_radio == settings.RTSP:
    helper.play_rtsp_stream(confidence, model)

elif source_radio == settings.YOUTUBE:
    helper.play_youtube_video(confidence, model)

else:
    st.error("Please select a valid source type!")