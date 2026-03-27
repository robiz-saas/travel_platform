import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import zipfile
import io
import os
import tempfile
import pandas as pd
from efficiency_predictor import EfficiencyPredictor  # NEW IMPORT

# === MODEL SETUP ===
CLASS_NAMES = ['bird_droppings', 'clean', 'dusty', 'electrical_damage', 'physical_damage', 'snow_covered']
MODEL_PATH = "../saved_models/resnet18_classifier.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# === NEW: Initialize efficiency predictor ===
@st.cache_resource
def load_efficiency_predictor():
    return EfficiencyPredictor()


efficiency_predictor = load_efficiency_predictor()


@st.cache_resource
def load_model():
    model = models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


model = load_model()

# === IMAGE TRANSFORM ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# === UPDATED: PREDICT FUNCTION WITH EFFICIENCY ===
def predict(image: Image.Image):
    """Original predict function - returns class and confidence"""
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        return CLASS_NAMES[predicted.item()], confidence.item()


# === NEW: COMPLETE PREDICT WITH EFFICIENCY ===
def predict_with_efficiency(image: Image.Image):
    """New function that returns classification + efficiency prediction"""
    # Get classification
    predicted_class, confidence = predict(image)

    # Get efficiency prediction
    efficiency_result = efficiency_predictor.predict_efficiency(
        defect_type=predicted_class,
        confidence=confidence
    )

    # Get recommendations
    recommendations = efficiency_predictor.get_recommendations(predicted_class)

    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'efficiency': efficiency_result,
        'recommendations': recommendations
    }


# === STREAMLIT UI ===
st.title("🔍 Solar Panel Defect Classifier & Efficiency Analyzer")
st.write("Upload either a **single image** or a **zip file** of images to classify defects and predict efficiency.")

# === NEW: Add toggle for efficiency analysis ===
include_efficiency = st.checkbox("📊 Include Efficiency Analysis", value=True,
                                 help="Get efficiency predictions along with defect classification")

option = st.radio("Choose input type:", ["Single Image", "Zip Folder (Batch Mode)"])

# === SINGLE IMAGE MODE ===
if option == "Single Image":
    uploaded_file = st.file_uploader("📤 Upload Single Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("🔎 Classify Image"):
            if include_efficiency:
                # NEW: Complete analysis with efficiency
                with st.spinner("Analyzing defects and predicting efficiency..."):
                    result = predict_with_efficiency(image)

                # Vertical single-column display
                st.subheader("📊 Analysis Results")

                # Single column layout with clear spacing
                st.metric(
                    "🔍 Defect Type",
                    result['predicted_class'].replace('_', ' ').title(),
                    help="Type of defect detected in the solar panel"
                )

                st.metric(
                    "📈 Confidence",
                    f"{result['confidence'] * 100:.1f}%",
                    help="Classification model confidence score"
                )

                if 'error' not in result['efficiency']:
                    st.metric(
                        "⚡ Efficiency",
                        f"{result['efficiency']['predicted_efficiency']}%",
                        help="Predicted solar panel efficiency"
                    )

                    priority = result['efficiency']['maintenance_priority']
                    # Color-code priority display
                    if priority == 'High':
                        st.error(f"🚨 Priority: **{priority}** - Immediate attention required")
                    elif priority == 'Medium':
                        st.warning(f"⚠️ Priority: **{priority}** - Schedule maintenance soon")
                    else:
                        st.success(f"✅ Priority: **{priority}** - Routine maintenance sufficient")
                else:
                    st.error("❌ Efficiency prediction error")

            else:
                # Original simple classification - vertical layout
                label, conf = predict(image)
                st.subheader("📊 Classification Results")

                st.metric(
                    "🔍 Defect Type",
                    label.replace('_', ' ').title(),
                    help="Type of defect detected"
                )
                st.metric(
                    "📈 Confidence",
                    f"{conf * 100:.1f}%",
                    help="Classification confidence score"
                )

# === BATCH MODE (ZIP FILE) ===
elif option == "Zip Folder (Batch Mode)":
    uploaded_zip = st.file_uploader("📤 Upload Folder of Images (.zip)", type=["zip"])

    if uploaded_zip:
        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_path = os.path.join(tmp_dir, "images.zip")
            with open(zip_path, "wb") as f:
                f.write(uploaded_zip.read())

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(tmp_dir)

            image_files = [os.path.join(tmp_dir, name) for name in zip_ref.namelist()
                           if name.lower().endswith(('.jpg', '.jpeg', '.png'))]

            if not image_files:
                st.warning("No valid images found in the uploaded zip.")
            else:
                st.success(f"Found {len(image_files)} images. Processing...")

                # NEW: Batch processing with efficiency
                if include_efficiency:
                    # Process all images and collect results
                    batch_results = []
                    progress_bar = st.progress(0)

                    for i, img_path in enumerate(image_files):
                        try:
                            image = Image.open(img_path).convert("RGB")
                            result = predict_with_efficiency(image)

                            # Collect data for summary table
                            if 'error' not in result['efficiency']:
                                batch_results.append({
                                    'Image': os.path.basename(img_path),
                                    'Defect Type': result['predicted_class'].replace('_', ' ').title(),
                                    'Confidence': f"{result['confidence'] * 100:.1f}%",
                                    'Efficiency': f"{result['efficiency']['predicted_efficiency']}%",
                                    'Priority': result['efficiency']['maintenance_priority']
                                })

                            # Display individual result - vertical layout
                            with st.expander(
                                    f"📊 {os.path.basename(img_path)} - {result['predicted_class'].replace('_', ' ').title()}"):
                                col_img, col_info = st.columns([1, 1])

                                with col_img:
                                    st.image(image, use_container_width=True)

                                with col_info:
                                    # Vertical info display
                                    st.write(f"**🔍 Defect Type:**")
                                    st.write(f"   {result['predicted_class'].replace('_', ' ').title()}")

                                    st.write(f"**📈 Confidence:**")
                                    st.write(f"   {result['confidence'] * 100:.1f}%")

                                    if 'error' not in result['efficiency']:
                                        st.write(f"**⚡ Efficiency:**")
                                        st.write(f"   {result['efficiency']['predicted_efficiency']}%")

                                        priority = result['efficiency']['maintenance_priority']
                                        st.write(f"**🔧 Priority:**")
                                        if priority == 'High':
                                            st.write(f"   🚨 **{priority}** (Urgent)")
                                        elif priority == 'Medium':
                                            st.write(f"   ⚠️ **{priority}** (Soon)")
                                        else:
                                            st.write(f"   ✅ **{priority}** (Routine)")

                        except Exception as e:
                            st.error(f"Error processing {img_path}: {e}")

                        progress_bar.progress((i + 1) / len(image_files))

                    # Summary table and download
                    if batch_results:
                        st.subheader("📋 Batch Analysis Summary")
                        df = pd.DataFrame(batch_results)
                        st.dataframe(df, use_container_width=True)

                        # Download results
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📄 Download Results (CSV)",
                            data=csv,
                            file_name="solar_panel_analysis.csv",
                            mime="text/csv"
                        )

                        # Summary statistics - vertical layout
                        st.subheader("📈 Summary Statistics")

                        avg_efficiency = df['Efficiency'].str.rstrip('%').astype(float).mean()
                        st.metric("📊 Average Efficiency", f"{avg_efficiency:.1f}%")

                        high_priority = len(df[df['Priority'] == 'High'])
                        st.metric("🚨 High Priority Issues", f"{high_priority} out of {len(df)} images")

                        most_common = df['Defect Type'].mode().iloc[0] if len(df) > 0 else "None"
                        st.metric("🔍 Most Common Defect", most_common)

                else:
                    # Original batch mode without efficiency
                    for img_path in image_files:
                        try:
                            image = Image.open(img_path).convert("RGB")
                            label, conf = predict(image)

                            # Show image with vertical info
                            col_img, col_info = st.columns([2, 1])

                            with col_img:
                                st.image(image, use_container_width=True)

                            with col_info:
                                st.write(f"**📁 File:** {os.path.basename(img_path)}")
                                st.write(f"**🔍 Defect:** {label.replace('_', ' ').title()}")
                                st.write(f"**📈 Confidence:** {conf * 100:.1f}%")

                        except Exception as e:
                            st.error(f"Error processing {img_path}: {e}")

# === Footer ===
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
<p><strong>🤖 Model:</strong> ResNet18 Classification + Statistical Efficiency Prediction</p>
<p><strong>📊 Data:</strong> Based on 603 real efficiency measurements across 6 defect categories</p>
</div>
""", unsafe_allow_html=True)