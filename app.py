import os
import sys
import numpy as np
import streamlit as st
from PIL import Image

# Ensure project root is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import WatermelonDiagnosticPipeline
from src.knowledge_base import DISEASE_KNOWLEDGE_BASE

# Streamlit translations
UI_TRANSLATIONS = {
    "English": {
        "title": "🍉 Noma AI - Watermelon Leaf Diagnostics",
        "subtitle": "Complete offline diagnostic pipeline. No cloud dependencies.",
        "input_header": "1. Source Image",
        "upload_label": "Upload leaf image (JPG, JPEG, PNG):",
        "camera_label": "Or take a photo using your device camera:",
        "results_header": "2. Diagnostic Results",
        "condition_label": "Predicted Condition:",
        "confidence_label": "Confidence Score:",
        "severity_header": "3. Severity & Recommendations",
        "severity_est_label": "Estimated Severity (Visual Proxy):",
        "severity_adjust_label": "Select/Adjust Severity Level:",
        "severity_warning_note": "Note: Severity is visually estimated by leaf spot ratios and is not a certified scientific prediction.",
        "rec_title": "Practical Recommendations",
        "details_title": "Disease Knowledge Base Info",
        "prob_title": "Detailed Probabilities",
        "quality_fail_warning": "⚠️ Image Quality Check Failed",
        "quality_bypass_label": "Bypass Quality Gate and diagnose anyway",
        "retake_msg": "We recommend retaking the photo in clearer conditions.",
        "disclaimer_title": "Important Disclaimer",
        "symptoms": "Symptoms",
        "causes": "Likely Causes",
        "prevention": "Prevention Measures",
        "management": "Management & Control",
        "guidance": "Farmer Guidance",
        "waiting": "Please upload or capture a watermelon leaf photo to start diagnostics.",
        "confidence_warning": "⚠️ Low Confidence prediction. The model is uncertain about this leaf's condition.",
        "classes": {
            "Anthracnose": "Anthracnose",
            "Downy_Mildew": "Downy Mildew",
            "Healthy": "Healthy",
            "Mosaic_Virus": "Mosaic Virus"
        }
    },
    "Hausa": {
        "title": "🍉 Noma AI - Gano Cututtukan Kankana",
        "subtitle": "Cikakken tsarin binciken cututtuka na offline. Ba ya bukatar intanet.",
        "input_header": "1. Hoton Ganye",
        "upload_label": "Dora hoton ganye (JPG, JPEG, PNG):",
        "camera_label": "Ko dauki hoton ganye da kyamara:",
        "results_header": "2. Sakamakon Bincike",
        "condition_label": "Ciwon da Aka Gano:",
        "confidence_label": "Matakin Tabbaci:",
        "severity_header": "3. Tsanani da Shawarwari",
        "severity_est_label": "Kiyasin Tsanani (Visual Proxy):",
        "severity_adjust_label": "Zaɓi/Gyara Matakin Tsanani:",
        "severity_warning_note": "Lura: An kiyasta matakin tsanani ne ta hanyar hoton tabo na ganye, ba binciken kimiyya na musamman ba ne.",
        "rec_title": "Shawarwari na Musamman",
        "details_title": "Bayanin Ciwo Dalla-dalla",
        "prob_title": "Cikakken Rarraba Samfuran Tabbaci",
        "quality_fail_warning": "⚠️ Gwajin Ingancin Hoto Bai Cika Ba",
        "quality_bypass_label": "Guji gargaɗin inganci kuma bincika hoton",
        "retake_msg": "Muna ba da shawarar sake daukar hoton a cikin haske mai kyau.",
        "disclaimer_title": "Gargaɗi mai Muhimmanci",
        "symptoms": "Alamomin Ciwo",
        "causes": "Abin da Ke Kawo Shi",
        "prevention": "Hanyoyin Rigakafi",
        "management": "Kula da Magani",
        "guidance": "Jagorar Manoma",
        "waiting": "Da fatan za a dora ko dauki hoton ganyen kankana don fara bincike.",
        "confidence_warning": "⚠️ Karancin Tabbaci. AI ba ta da cikakken yakinin wannan binciken.",
        "classes": {
            "Anthracnose": "Ciwon Anthracnose (Digo-digo a Ganye)",
            "Downy_Mildew": "Ciwon Downy Mildew (Farin Kura a Ganye)",
            "Healthy": "Lafiyayyen Ganye (Bashi da Ciwo)",
            "Mosaic_Virus": "Ciwon Mosaic Virus (Kankana mai Gurgurar Ganye)"
        }
    }
}

@st.cache_resource
def load_diagnostic_pipeline():
    """Loads and caches the central pipeline to avoid reloading the TFLite model on rerun."""
    return WatermelonDiagnosticPipeline()

def main():
    # Set page configuration
    st.set_page_config(
        page_title="Noma AI Diagnostics",
        page_icon="🍉",
        layout="centered"
    )
    
    # 1. Language selector in sidebar
    lang = st.sidebar.selectbox(
        "Language / Yare",
        options=["English", "Hausa"],
        index=0
    )
    
    t = UI_TRANSLATIONS[lang]
    
    st.title(t["title"])
    st.markdown(t["subtitle"])
    st.divider()
    
    # Load pipeline
    try:
        pipeline = load_diagnostic_pipeline()
    except Exception as e:
        st.error(f"Error loading pipeline: {e}")
        return
        
    # Input area
    st.subheader(t["input_header"])
    uploaded_file = st.file_uploader(t["upload_label"], type=["jpg", "jpeg", "png"])
    camera_file = st.camera_input(t["camera_label"])
    
    active_file = camera_file if camera_file is not None else uploaded_file
    
    if active_file is not None:
        try:
            # Load and display image
            image = Image.open(active_file)
            st.image(image, caption="Active Image", use_container_width=True)
            
            # Analyze image quality first (without running inference yet)
            quality_results = pipeline.quality_gate.analyze_image(image)
            
            bypass_quality = False
            
            if not quality_results["passed"]:
                st.warning(t["quality_fail_warning"])
                
                # Show individual quality warnings
                warning_text = pipeline.rec_engine.get_quality_warning(quality_results["reasons"], lang=lang)
                st.write(warning_text)
                
                # Option to override quality gate
                bypass_quality = st.checkbox(t["quality_bypass_label"], value=False)
                
                if not bypass_quality:
                    st.info(t["retake_msg"])
                    
            # Run diagnostics if quality passes OR bypass is checked
            if quality_results["passed"] or bypass_quality:
                with st.spinner("Analyzing Leaf / Ana Bincike..."):
                    # First run diagnose to get automatic estimates
                    res = pipeline.diagnose(
                        image, 
                        bypass_quality_gate=True, # bypassed because we already checked it and got consent
                        lang=lang
                    )
                    
                st.divider()
                st.header(t["results_header"])
                
                # Handle low-confidence prediction warnings
                is_low_conf = (res["confidence_status"] == "LOW_CONFIDENCE")
                if is_low_conf:
                    st.warning(t["confidence_warning"])
                    st.info(res["recommendation_result"]["recommendation"])
                
                # Columns for Condition and Confidence
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader(t["condition_label"])
                    # If low confidence, show uncertain label
                    cond_display = res["diagnosis_translated"]
                    if is_low_conf:
                        st.error(f"**{cond_display}**")
                    else:
                        st.success(f"**{cond_display}**")
                        
                with col2:
                    st.subheader(t["confidence_label"])
                    st.metric(label="", value=f"{res['confidence_score'] * 100:.2f}%")
                    
                # Show details if prediction is high confidence
                if not is_low_conf:
                    st.divider()
                    st.header(t["severity_header"])
                    
                    # Estimate display
                    est_severity = res["severity"]  # Estimated initially by proxy
                    
                    # User selector for severity override
                    severity_opts = ["Low", "Moderate", "High"] if lang == "English" else ["Low", "Moderate", "High"]
                    # Map severe options
                    severity_idx = severity_opts.index(est_severity)
                    
                    selected_severity = st.selectbox(
                        t["severity_adjust_label"],
                        options=severity_opts,
                        index=severity_idx
                    )
                    
                    st.caption(t["severity_warning_note"])
                    
                    # Fetch recommendation based on the selected severity
                    rec_result = pipeline.rec_engine.get_recommendation(
                        diagnosis=res["diagnosis"],
                        confidence_status=res["confidence_status"],
                        severity=selected_severity,
                        lang=lang
                    )
                    
                    # Display recommendation
                    st.subheader(t["rec_title"])
                    st.info(rec_result["recommendation"])
                    
                    # Display Knowledge Base entries
                    st.subheader(t["details_title"])
                    info = res["disease_info"]
                    if info:
                        with st.expander(t["symptoms"]):
                            st.write(info["symptoms"])
                        with st.expander(t["causes"]):
                            st.write(info["causes"])
                        with st.expander(t["prevention"]):
                            st.write(info["prevention"])
                        with st.expander(t["management"]):
                            st.write(info["management"])
                        with st.expander(t["guidance"]):
                            st.write(info["farmer_guidance"])
                            
                # Detailed Probabilities
                st.divider()
                st.subheader(t["prob_title"])
                for class_raw, score in res["predictions_breakdown"].items():
                    class_display = t["classes"].get(class_raw, class_raw)
                    st.write(f"{class_display}: {score * 100:.2f}%")
                    st.progress(float(score))
                    
        except Exception as e:
            st.error(f"Error processing image: {e}")
            st.exception(e)
    else:
        st.info(t["waiting"])
        
    # Footer Disclaimer
    st.divider()
    st.caption(f"**{t['disclaimer_title']}**")
    st.caption(DISEASE_KNOWLEDGE_BASE["Disclaimer"][lang])

if __name__ == "__main__":
    main()
