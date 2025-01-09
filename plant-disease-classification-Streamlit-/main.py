import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import base64
import json


def model_prediction(test_image):
    model = tf.keras.models.load_model('trained_model.keras')
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    predictions = model.predict(input_arr)
    return np.argmax(predictions, axis=1)[0], np.max(predictions) * 100

def load_class_names():
    with open('class_names.json', 'r') as file:
        data = json.load(file)
    return data['class_names']


def get_base64(file_path):
    """Function to encode file to base64"""
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background_video(video_file):
    """Function to set background video using local file"""
    video_base64 = get_base64(video_file)
    
    # CSS and HTML to handle video background
    st.markdown(
        f'''
        <style>
            /* Video background styles */
            .stApp {{
                background: transparent;
            }}
            
            .video-background {{
                position: fixed;
                top: 0;
                left: 0;
                width: 100vw;
                height: 100vh;
                overflow: hidden;
                z-index: -1;
            }}
            
            #myVideo {{
                position: absolute;
                right: 0;
                bottom: 0;
                min-width: 100%;
                min-height: 100%;
                width: auto;
                height: auto;
            }}

            /* Ensure Markdown content remains visible */
            .stMarkdown {{
                background-color: rgba(255, 255, 255, 0.7);
                padding: 20px;
                border-radius: 5px;
                margin: 10px 0;
            }}

            /* Style headers */
            .stMarkdown h1,
            .stMarkdown h2,
            .stMarkdown h3 {{
                color: rgb(38, 39, 48);
            }}

            /* Style paragraphs */
            .stMarkdown p {{
                color: rgb(38, 39, 48);
            }}
        </style>
        <div class="video-background">
            <video autoplay muted loop playsinline id="myVideo">
                <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
            </video>
        </div>
        ''',
        unsafe_allow_html=True
    )

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.radio("Select Page", ["Home", "Disease Recognition","About"])
video_file = "itachi-walk.mp4"  # Video file path
set_background_video(video_file)
# Main Page
if app_mode == "Home":
    st.header("PLANT DISEASE CLASSIFICATION SYSTEM")

    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

# About Project
elif app_mode == "About":
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this GitHub repo.
                This dataset consists of about 87K RGB images of healthy and diseased crop leaves which are categorized into 38 different classes. The total dataset is divided into an 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purposes.
                #### Content
                1. train (70295 images)
                2. test (33 images)
                3. validation (17572 images)
                """)

# Prediction Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if test_image is not None:
        if st.button("Show Image"):
            st.image(test_image)
        # Predict button
        if st.button("Predict"):
            st.write("Our Prediction")
            result_index,confidence = model_prediction(test_image)
            # Reading Labels
            class_name = load_class_names()
            st.write(class_name[result_index])
             # Add visual confidence indicator
            if confidence > 90:
                st.success(f"High confidence prediction: {confidence:.2f}%")
            elif confidence > 70:
                st.info(f"Moderate confidence prediction: {confidence:.2f}%")
            else:
                st.warning(f"Low confidence prediction: {confidence:.2f}%")