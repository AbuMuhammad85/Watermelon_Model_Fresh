import os
import numpy as np
import tensorflow as tf
from PIL import Image
import streamlit as st

# Class names in alphabetical order as trained
CLASSES = ["Anthracnose", "Downy_Mildew", "Healthy", "Mosaic_Virus"]

# English and Hausa translations
TRANSLATIONS = {
    "English": {
        "title": "Watermelon Leaf Disease Diagnostics",
        "subtitle": "Classify watermelon plant leaves using an offline MobileNetV3-Small TFLite model.",
        "lang_label": "Select Language / Zaɓi Yare",
        "input_section": "Input Sources",
        "upload_lbl": "Upload leaf image (JPG, JPEG, PNG):",
        "camera_lbl": "Or take a photo using your camera:",
        "results_title": "Diagnostic Results",
        "predicted_disease": "Predicted Condition:",
        "confidence": "Confidence Score:",
        "prob_dist": "Class Probability Breakdown",
        "waiting": "Please upload an image or capture a photo to view diagnostics.",
        "class_names": {
            "Anthracnose": "Anthracnose",
            "Downy_Mildew": "Downy Mildew",
            "Healthy": "Healthy",
            "Mosaic_Virus": "Mosaic Virus"
        }
    },
    "Hausa": {
        "title": "Gano Cututtukan Ganyen Kankana",
        "subtitle": "Binciki lafiyar ganyen kankana ta amfani da model na MobileNetV3-Small TFLite offline.",
        "lang_label": "Select Language / Zaɓi Yare",
        "input_section": "Hanyar Samun Hoto",
        "upload_lbl": "Dora hoton ganye (JPG, JPEG, PNG):",
        "camera_lbl": "Ko dauki hoton ganye da kyamara:",
        "results_title": "Sakamakon Bincike",
        "predicted_disease": "Ciwon da Aka Gano:",
        "confidence": "Matakin Tabbaci:",
        "prob_dist": "Rarraba Samfuran Tabbaci",
        "waiting": "Da fatan za a dora hoto ko dauki hoto don fara bincike.",
        "class_names": {
            "Anthracnose": "Ciwon Anthracnose (Digo-digo a Ganye)",
            "Downy_Mildew": "Ciwon Downy Mildew (Farin Kura a Ganye)",
            "Healthy": "Lafiyayyen Ganye (Bashi da Ciwo)",
            "Mosaic_Virus": "Ciwon Mosaic Virus (Kankana mai Gurgurar Ganye)"
        }
    }
}

@st.cache_resource
def load_tflite_interpreter():
    """Loads the TFLite interpreter and caches it to prevent reloading on each rerun."""
    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(workspace_dir, "checkpoints", "model_float16.tflite")
    
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def run_inference(interpreter, pil_image):
    """Preprocesses the image and runs inference through the TFLite model."""
    # Preprocess image to match training pipeline (RGB, 224x224, float32, range [0.0, 255.0])
    img = pil_image.convert("RGB")
    img = img.resize((224, 224), Image.Resampling.BILINEAR)
    
    input_data = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    
    predictions = interpreter.get_tensor(output_details[0]["index"])[0]
    return predictions

def main():
    # Set page configuration
    st.set_page_config(
        page_title="Watermelon Disease Diagnostics",
        page_icon="🍉",
        layout="centered"
    )
    
    # 1. Language selector
    lang = st.sidebar.selectbox(
        "Select Language / Zaɓi Yare",
        options=["English", "Hausa"],
        index=0
    )
    
    t = TRANSLATIONS[lang]
    
    # Title & Header
    st.title(t["title"])
    st.markdown(t["subtitle"])
    st.divider()
    
    # Load TFLite Model
    try:
        interpreter = load_tflite_interpreter()
    except Exception as e:
        st.error(f"Error loading TFLite model: {e}")
        return
        
    # Input options: Upload or Camera
    st.subheader(t["input_section"])
    uploaded_file = st.file_uploader(t["upload_lbl"], type=["jpg", "jpeg", "png"])
    camera_file = st.camera_input(t["camera_lbl"])
    
    # Process image if available (camera input takes precedence if both are active)
    active_file = camera_file if camera_file is not None else uploaded_file
    
    if active_file is not None:
        # Load and display image
        try:
            image = Image.open(active_file)
            st.image(image, caption=active_file.name if hasattr(active_file, "name") else "Captured Image", use_container_width=True)
            
            # Run TFLite Prediction
            with st.spinner("Analyzing / Ana Bincike..."):
                preds = run_inference(interpreter, image)
                
            class_idx = np.argmax(preds)
            pred_class_raw = CLASSES[class_idx]
            pred_class_display = t["class_names"][pred_class_raw]
            confidence = preds[class_idx]
            
            # Display Results
            st.divider()
            st.header(t["results_title"])
            
            # Show predicted disease and confidence
            col1, col2 = st.columns(2)
            with col1:
                st.subheader(t["predicted_disease"])
                st.info(f"**{pred_class_display}**")
            with col2:
                st.subheader(t["confidence"])
                st.metric(label="", value=f"{confidence * 100:.2f}%")
                
            # Class probabilities breakdown
            st.subheader(t["prob_dist"])
            for cls_raw, score in zip(CLASSES, preds):
                cls_display = t["class_names"][cls_raw]
                st.write(f"{cls_display}: {score * 100:.2f}%")
                st.progress(float(score))
                
        except Exception as e:
            st.error(f"Error processing image: {e}")
    else:
        st.info(t["waiting"])

if __name__ == "__main__":
    main()
