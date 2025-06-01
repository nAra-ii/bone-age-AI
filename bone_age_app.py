import streamlit as st
import datetime
import os
from PIL import Image
import numpy as np
import tensorflow as tf

# Title
st.title("🦴 Bone Age App - Greulich & Pyle + AI Assist")

# Patient Details
st.header("1. Patient Information")
dob = st.date_input("Date of Birth")
xray_date = st.date_input("Date of X-ray", datetime.date.today())
sex = st.selectbox("Sex", ["male", "female"])

# Calculate Chronological Age
if xray_date and dob:
    age_days = (xray_date - dob).days
    years = age_days // 365
    months = (age_days % 365) // 30
    st.success(f"Chronological Age: {years} years {months} months")
    age_label = f"{sex}_{years}_{months}"  # used for filename matching

# Upload X-ray
st.header("2. Upload Hand X-ray")
xray_file = st.file_uploader("Upload hand radiograph (JPEG/PNG)", type=["jpg", "jpeg", "png"])

# Load G&P Atlas Images
st.header("3. Atlas Matching")
atlas_dir = "gp_atlas"  # your folder containing all G&P images
atlas_files = sorted(os.listdir(atlas_dir)) if os.path.exists(atlas_dir) else []

# AI Prediction Helper
@st.cache_resource(allow_output_mutation=True)
def load_model():
    # This should point to a simple pre-trained Keras model you upload
    return tf.keras.models.load_model("bone_age_model.h5")

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    if image.shape[-1] == 4:
        image = image[..., :3]  # remove alpha if exists
    image = np.expand_dims(image, axis=0)
    return image

selected_file = None
if xray_file and atlas_files:
    xray_img = Image.open(xray_file)

    # AI Bone Age Prediction
    st.subheader("AI-Assisted Bone Age Suggestion")
    model = load_model()
    input_img = preprocess_image(xray_img)
    predicted_months = model.predict(input_img)[0][0]
    predicted_years = int(predicted_months // 12)
    predicted_month = int(predicted_months % 12)
    st.info(f"📊 AI Predicted Bone Age: {predicted_years} years {predicted_month} months")

    # Match with closest G&P plate
    def age_diff(file):
        parts = file.replace(".png", "").replace(".jpg", "").split("_")
        if parts[0] != sex:
            return float('inf')
        y, m = int(parts[1]), int(parts[2])
        return abs((y * 12 + m) - predicted_months)

    atlas_files_filtered = [f for f in atlas_files if f.startswith(sex)]
    closest_file = min(atlas_files_filtered, key=age_diff)
    selected_file = closest_file

    st.subheader("AI-Suggested Atlas Match")
    col1, col2 = st.columns(2)
    with col1:
        st.image(xray_img, caption="Uploaded X-ray", use_column_width=True)
    with col2:
        ref_img = Image.open(os.path.join(atlas_dir, closest_file))
        st.image(ref_img, caption=f"Suggested: {closest_file.replace('_', ' ').replace('.png','')}", use_column_width=True)

    # Manual scroll
    st.subheader("Manual Atlas Browsing")
    selected_file = st.selectbox("Select closest match manually (optional)", atlas_files_filtered, index=atlas_files_filtered.index(closest_file))

# SD Selection
st.header("4. Standard Deviation and Comments")
sd = st.selectbox("Standard Deviation (SD)", ["-2.0", "-1.5", "-1.0", "-0.5", "0", "+0.5", "+1.0", "+1.5", "+2.0"])
comment = st.text_area("Comments (optional)")

# Export Result
st.header("5. Summary")
if selected_file:
    bone_age_str = selected_file.replace(".png", "").replace(".jpg", "").replace("_", " ")
    summary = f"""
    🧾 **Bone Age Report**
    - Chronological Age: {years} years {months} months
    - Sex: {sex.title()}
    - AI Predicted Bone Age: {predicted_years} years {predicted_month} months
    - Bone Age Match: {bone_age_str} (Greulich & Pyle)
    - Standard Deviation: {sd}
    - Comments: {comment if comment else '-'}
    """
    st.markdown(summary)
    st.download_button("📥 Download Report", summary, file_name="bone_age_report.txt")
