import streamlit as st
import torch
from PIL import Image
import zipfile
import tempfile
import os
import pandas as pd
import io
from datetime import datetime
from ai_predict import AISolarPanelPredictor
from pdf_generator import SolarPanelPDFGenerator


@st.cache_resource
def load_pdf_generator():
    """Load PDF generator (cached for performance)"""
    return SolarPanelPDFGenerator()


@st.cache_resource
def load_pdf_generator():
    """Load PDF generator (cached for performance)"""
    return SolarPanelPDFGenerator()


# Page configuration
st.set_page_config(
    page_title="AI Solar Panel Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = None
if 'pdf_ready' not in st.session_state:
    st.session_state.pdf_ready = False
if 'pdf_data' not in st.session_state:
    st.session_state.pdf_data = None
if 'pdf_filename' not in st.session_state:
    st.session_state.pdf_filename = None

# Custom CSS for better styling
st.markdown("""
<style>
.result-box {
    background-color: #f0f2f6;
    padding: 20px;
    border-radius: 10px;
    margin-top: 10px;
    border-left: 5px solid #3b82f6;
}
.pdf-section {
    background-color: #e8f4fd;
    padding: 15px;
    border-radius: 8px;
    margin-top: 15px;
    border-left: 4px solid #1976d2;
}
.batch-result {
    background-color: #f8f9fa;
    padding: 10px;
    border-radius: 5px;
    margin: 5px 0;
    border-left: 3px solid #28a745;
}
</style>
""", unsafe_allow_html=True)


# Load the AI model
@st.cache_resource
def load_ai_predictor():
    model_path = "saved_models/best_multi_output_model.pth"
    if not os.path.exists(model_path):
        st.error("AI model not found. Please train the model first.")
        return None
    return AISolarPanelPredictor(model_path)


def display_prediction(result, show_pdf_option=True, result_key="single"):
    """Display prediction results with optional PDF generation"""
    if not result['success']:
        st.error(f"Prediction failed: {result['error']}")
        return

    st.markdown("<div class='result-box'>", unsafe_allow_html=True)

    # Create columns for better layout
    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**🔍 Defect Type:** {result['predicted_class'].replace('_', ' ').title()}")
        st.write(f"**🎯 Confidence:** {result['confidence'] * 100:.2f}%")

    with col2:
        st.write(f"**⚡ Efficiency:** {result['predicted_efficiency']:.1f}%")

        # Priority with color coding
        priority = result['priority']
        if priority == 'High':
            st.write(f"**🚨 Priority:** <span style='color: red; font-weight: bold;'>{priority}</span>",
                     unsafe_allow_html=True)
        elif priority == 'Medium':
            st.write(f"**⚠️ Priority:** <span style='color: orange; font-weight: bold;'>{priority}</span>",
                     unsafe_allow_html=True)
        else:
            st.write(f"**✅ Priority:** <span style='color: green; font-weight: bold;'>{priority}</span>",
                     unsafe_allow_html=True)

    # Recommendations
    if 'recommendations' in result and result['recommendations']:
        st.write("**💡 Recommendations:**")
        for i, rec in enumerate(result['recommendations'][:3], 1):  # Show top 3 recommendations
            st.write(f"   {i}. {rec}")

    st.markdown("</div>", unsafe_allow_html=True)

    # PDF Generation Option
    if show_pdf_option:
        st.markdown("<div class='pdf-section'>", unsafe_allow_html=True)
        st.write("**📄 Generate Detailed PDF Report**")

        col_pdf1, col_pdf2, col_pdf3 = st.columns([1, 1, 2])

        with col_pdf1:
            if st.button("📄 Generate PDF", key=f"generate_pdf_{result_key}", type="secondary"):
                with st.spinner("Generating PDF report..."):
                    generate_pdf_data(result, is_batch=(result_key == "batch"))

        with col_pdf2:
            # Show download button if PDF is ready
            if st.session_state.pdf_ready and st.session_state.pdf_data:
                st.download_button(
                    label="📥 Download PDF",
                    data=st.session_state.pdf_data,
                    file_name=st.session_state.pdf_filename,
                    mime="application/pdf",
                    key=f"download_pdf_{result_key}",
                    help="Download comprehensive analysis report as PDF"
                )

        with col_pdf3:
            if st.session_state.pdf_ready:
                st.success("✅ PDF ready for download!")
            else:
                st.info("💡 PDF includes detailed analysis, charts, and recommendations")

        st.markdown("</div>", unsafe_allow_html=True)


def generate_pdf_data_simple(analysis_result, is_batch=False):
    """Simple PDF generation without complex temp file handling"""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        from reportlab.lib.units import inch

        # Create temp file for PDF output
        temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        temp_pdf_path = temp_pdf.name
        temp_pdf.close()

        # Create PDF document
        doc = SimpleDocTemplate(
            temp_pdf_path,
            pagesize=letter,
            rightMargin=72, leftMargin=72,
            topMargin=72, bottomMargin=72
        )

        # Story elements
        story = []
        styles = getSampleStyleSheet()

        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1,  # Center
            textColor=colors.HexColor('#1f4e79')
        )

        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            textColor=colors.HexColor('#2c5f2d')
        )

        if is_batch:
            # Batch report
            story.append(Paragraph("Solar Panel Batch Analysis Report", title_style))
            story.append(Spacer(1, 30))

            # Summary stats
            total_panels = len(analysis_result)
            avg_efficiency = sum(
                r.get('predicted_efficiency', 0) for r in analysis_result) / total_panels if total_panels > 0 else 0
            high_priority = sum(1 for r in analysis_result if r.get('priority') == 'High')

            story.append(Paragraph("Executive Summary", heading_style))
            story.append(Paragraph(f"<b>Total Panels Analyzed:</b> {total_panels}", styles['Normal']))
            story.append(Paragraph(f"<b>Average Efficiency:</b> {avg_efficiency:.1f}%", styles['Normal']))
            story.append(Paragraph(f"<b>High Priority Issues:</b> {high_priority}", styles['Normal']))
            story.append(Spacer(1, 20))

            # Individual results
            story.append(Paragraph("Detailed Results", heading_style))
            for i, result in enumerate(analysis_result[:20], 1):  # Limit to first 20
                image_name = result.get('image_name', f'Panel_{i}')
                predicted_class = result.get('predicted_class', 'Unknown')
                confidence = result.get('confidence', 0)
                efficiency = result.get('predicted_efficiency', 0)
                priority = result.get('priority', 'Unknown')

                story.append(Paragraph(f"<b>Panel {i}: {image_name}</b>", styles['Heading3']))
                story.append(Paragraph(f"Defect Type: {predicted_class.replace('_', ' ').title()}", styles['Normal']))
                story.append(Paragraph(f"Confidence: {confidence * 100:.1f}%", styles['Normal']))
                story.append(Paragraph(f"Efficiency: {efficiency:.1f}%", styles['Normal']))
                story.append(Paragraph(f"Priority: {priority}", styles['Normal']))
                story.append(Spacer(1, 15))

            if len(analysis_result) > 20:
                story.append(Paragraph(f"<i>... and {len(analysis_result) - 20} more panels</i>", styles['Normal']))

        else:
            # Single panel report
            story.append(Paragraph("Solar Panel Analysis Report", title_style))
            story.append(Spacer(1, 30))

            # Panel info
            if 'image_name' in analysis_result:
                story.append(Paragraph(f"<b>Panel:</b> {analysis_result['image_name']}", styles['Normal']))
                story.append(Spacer(1, 10))

            # Analysis date
            story.append(
                Paragraph(f"<b>Analysis Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
            story.append(Spacer(1, 20))

            # Results section
            story.append(Paragraph("Analysis Results", heading_style))

            defect_type = analysis_result.get('predicted_class', 'Unknown').replace('_', ' ').title()
            story.append(Paragraph(f"<b>Defect Type:</b> {defect_type}", styles['Normal']))
            story.append(Spacer(1, 8))

            confidence = analysis_result.get('confidence', 0) * 100
            story.append(Paragraph(f"<b>Confidence Level:</b> {confidence:.2f}%", styles['Normal']))
            story.append(Spacer(1, 8))

            efficiency = analysis_result.get('predicted_efficiency', 0)
            story.append(Paragraph(f"<b>Predicted Efficiency:</b> {efficiency:.1f}%", styles['Normal']))
            story.append(Spacer(1, 8))

            priority = analysis_result.get('priority', 'Unknown')
            priority_color = '#d32f2f' if priority == 'High' else '#f57c00' if priority == 'Medium' else '#388e3c'
            story.append(Paragraph(f"<b>Maintenance Priority:</b> <font color='{priority_color}'>{priority}</font>",
                                   styles['Normal']))
            story.append(Spacer(1, 20))

            # Recommendations section
            if 'recommendations' in analysis_result and analysis_result['recommendations']:
                story.append(Paragraph("Recommendations", heading_style))
                for i, rec in enumerate(analysis_result['recommendations'], 1):
                    story.append(Paragraph(f"{i}. {rec}", styles['Normal']))
                    story.append(Spacer(1, 5))
                story.append(Spacer(1, 20))

            # Footer
            story.append(Spacer(1, 40))
            story.append(Paragraph("Generated by AI Solar Panel Analyzer", styles['Italic']))

        # Build PDF
        doc.build(story)

        # Read PDF data
        with open(temp_pdf_path, 'rb') as f:
            pdf_data = f.read()

        # Clean up temp file
        os.unlink(temp_pdf_path)

        # Store in session state
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if is_batch:
            filename = f"solar_batch_report_{timestamp}.pdf"
        else:
            filename = f"solar_analysis_report_{timestamp}.pdf"

        st.session_state.pdf_data = pdf_data
        st.session_state.pdf_filename = filename
        st.session_state.pdf_ready = True

        return True

    except Exception as e:
        st.error(f"❌ Simple PDF generation failed: {str(e)}")
        st.session_state.pdf_ready = False
        return False


def generate_pdf_data(analysis_result, is_batch=False):
    """Generate PDF report and store in session state - with improved error handling"""
    try:
        pdf_generator = load_pdf_generator()

        if is_batch:
            # For batch processing - analysis_result is a list of results
            if not analysis_result or len(analysis_result) == 0:
                st.error("No batch results to generate PDF")
                return False

            st.info(f"Generating batch PDF for {len(analysis_result)} results...")

            try:
                pdf_path = pdf_generator.generate_batch_analysis_report(analysis_result)
            except Exception as batch_error:
                st.warning(f"Original batch PDF generator failed: {str(batch_error)}")
                st.info("🔄 Trying simplified batch PDF generation...")
                return generate_pdf_data_simple(analysis_result, is_batch=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"solar_batch_report_{timestamp}.pdf"

        else:
            # For single image - analysis_result is a single result dict
            analysis_result_copy = analysis_result.copy()

            # Add image name
            if st.session_state.uploaded_image and hasattr(st.session_state.uploaded_image, 'name'):
                analysis_result_copy['image_name'] = st.session_state.uploaded_image.name
            else:
                analysis_result_copy['image_name'] = "solar_panel_image.jpg"

            # Create temp image file if image exists
            temp_image_path = None
            if st.session_state.uploaded_image:
                try:
                    # Create temp file for image
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                        temp_image_path = tmp_file.name

                        # Get image data and convert to PIL Image
                        if hasattr(st.session_state.uploaded_image, 'getvalue'):
                            # It's a Streamlit UploadedFile
                            image_bytes = st.session_state.uploaded_image.getvalue()
                            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                        else:
                            # It's already a PIL Image
                            image = st.session_state.uploaded_image.convert("RGB")

                        # Save to temp file
                        image.save(tmp_file, format='JPEG', quality=85)

                    # Debug info
                    st.info(f"Created temp image at: {temp_image_path}, size: {os.path.getsize(temp_image_path)} bytes")

                except Exception as img_error:
                    st.warning(f"Could not process image: {str(img_error)}. Generating PDF without image.")
                    temp_image_path = None

            # Generate PDF
            pdf_path = pdf_generator.generate_single_analysis_report(
                analysis_result_copy,
                image_path=temp_image_path
            )

            # Clean up temp image
            if temp_image_path and os.path.exists(temp_image_path):
                try:
                    os.remove(temp_image_path)
                except:
                    pass  # Ignore cleanup errors

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"solar_analysis_report_{timestamp}.pdf"

        # Read PDF file and store in session state
        if pdf_path and os.path.exists(pdf_path):
            with open(pdf_path, 'rb') as pdf_file:
                st.session_state.pdf_data = pdf_file.read()

            st.session_state.pdf_filename = filename
            st.session_state.pdf_ready = True

            # Clean up generated PDF file (it's now in memory)
            try:
                os.remove(pdf_path)
            except:
                pass  # Ignore cleanup errors

            # Clean up PDF generator temp files
            pdf_generator.cleanup_temp_files()

            if is_batch:
                st.success(f"✅ Batch PDF generated successfully for {len(analysis_result)} panels!")
            else:
                st.success("✅ PDF generated successfully with image and charts!")

            return True
        else:
            raise FileNotFoundError(f"Generated PDF file not found at {pdf_path}")

    except Exception as e:
        st.error(f"❌ PDF generation failed: {str(e)}")
        st.session_state.pdf_ready = False

        # Fallback to simple PDF generation for both single and batch
        st.info("🔄 Trying simplified PDF generation...")
        return generate_pdf_data_simple(analysis_result, is_batch)


def reset_pdf_state():
    """Reset PDF-related session state"""
    st.session_state.pdf_ready = False
    st.session_state.pdf_data = None
    st.session_state.pdf_filename = None


def main():
    st.title("🔍 AI Solar Panel Analyzer with PDF Reports")
    st.markdown("**AI-Powered Defect Detection & Efficiency Prediction**")

    # Sidebar information
    with st.sidebar:
        st.header("📊 About This Tool")
        st.info("""
        This AI system analyzes solar panels to:
        • 🔍 Detect defect types
        • ⚡ Predict efficiency levels  
        • 🎯 Assess maintenance priority
        • 💡 Provide recommendations
        """)

        st.header("📄 PDF Reports Include")
        st.success("""
        • Detailed analysis results
        • Visual efficiency gauges
        • Priority indicators
        • Maintenance recommendations
        • Technical details
        • Professional formatting
        """)

        # Clear results button
        if st.button("🗑️ Clear All Results", type="secondary"):
            st.session_state.analysis_results = None
            st.session_state.uploaded_image = None
            st.session_state.batch_results = None
            reset_pdf_state()
            # Clear cached PDF generator
            st.cache_resource.clear()
            st.rerun()

    predictor = load_ai_predictor()
    if predictor is None:
        st.stop()

    # Mode selection
    mode = st.radio("📋 Choose Analysis Mode", ["Single Image Analysis", "Batch Processing (ZIP Upload)"])

    if mode == "Single Image Analysis":
        st.header("📸 Single Image Analysis")

        uploaded_file = st.file_uploader(
            "📤 Upload Solar Panel Image",
            type=["jpg", "jpeg", "png"],
            help="Upload a single solar panel image for AI analysis"
        )

        if uploaded_file:
            # Store uploaded file in session state
            st.session_state.uploaded_image = uploaded_file

            # Reset PDF state when new image is uploaded
            if st.session_state.analysis_results is None:
                reset_pdf_state()

            # Display image
            image = Image.open(uploaded_file).convert("RGB")

            col_img, col_info = st.columns([2, 1])

            with col_img:
                st.image(image, caption=f"📁 {uploaded_file.name}", use_container_width=True)

            with col_info:
                st.subheader("📋 Image Info")
                st.write(f"**Filename:** {uploaded_file.name}")
                st.write(f"**Size:** {image.size}")
                st.write(f"**Format:** {image.format}")

            # Analysis button
            if st.button("🔍 Analyze with AI", type="primary", use_container_width=True):
                with st.spinner("🤖 AI is analyzing the image... Please wait"):
                    result = predictor.predict(image)

                    if result['success']:
                        # Store results in session state
                        st.session_state.analysis_results = result
                        reset_pdf_state()  # Reset PDF state for new analysis
                        st.success("✅ AI analysis completed successfully!")
                    else:
                        st.error(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")

        # Display stored results if available
        if st.session_state.analysis_results:
            display_prediction(st.session_state.analysis_results, show_pdf_option=True, result_key="single")

    elif mode == "Batch Processing (ZIP Upload)":
        st.header("📁 Batch Processing")

        uploaded_zip = st.file_uploader(
            "📤 Upload ZIP File with Solar Panel Images",
            type=["zip"],
            help="Upload a ZIP file containing multiple solar panel images"
        )

        if uploaded_zip:
            with tempfile.TemporaryDirectory() as tmp_dir:
                # Extract ZIP file
                zip_path = os.path.join(tmp_dir, "images.zip")
                with open(zip_path, "wb") as f:
                    f.write(uploaded_zip.read())

                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(tmp_dir)

                # Find image files
                image_paths = []
                for root, _, files in os.walk(tmp_dir):
                    for file in files:
                        if file.lower().endswith((".jpg", ".jpeg", ".png")):
                            image_paths.append(os.path.join(root, file))

                if not image_paths:
                    st.warning("❌ No valid images found in the ZIP file.")
                else:
                    st.success(f"✅ Found {len(image_paths)} images. Starting batch analysis...")

                    # Processing options
                    col_opt1, col_opt2 = st.columns(2)

                    with col_opt1:
                        show_images = st.checkbox("Show individual images", value=True)
                        max_display = st.slider("Max images to display", 1, min(20, len(image_paths)),
                                                min(10, len(image_paths)))

                    with col_opt2:
                        auto_process = st.checkbox("Auto-process all images", value=False)

                    # Start processing
                    if st.button("🚀 Start Batch Analysis", type="primary") or auto_process:
                        batch_results = []
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        # Process images
                        for i, path in enumerate(image_paths):
                            try:
                                status_text.text(f"Processing {i + 1}/{len(image_paths)}: {os.path.basename(path)}")

                                image = Image.open(path).convert("RGB")
                                result = predictor.predict(image)

                                if result['success']:
                                    # Add filename to result
                                    result['image_name'] = os.path.basename(path)
                                    batch_results.append(result)

                                    # Show individual results (limited)
                                    if show_images and i < max_display:
                                        with st.expander(
                                                f"📊 {os.path.basename(path)} - {result['predicted_class'].replace('_', ' ').title()}"):
                                            col_batch_img, col_batch_result = st.columns([1, 2])

                                            with col_batch_img:
                                                st.image(image, use_container_width=True)

                                            with col_batch_result:
                                                st.markdown("<div class='batch-result'>", unsafe_allow_html=True)
                                                st.write(
                                                    f"**Defect:** {result['predicted_class'].replace('_', ' ').title()}")
                                                st.write(f"**Confidence:** {result['confidence'] * 100:.1f}%")
                                                st.write(f"**Efficiency:** {result['predicted_efficiency']:.1f}%")
                                                st.write(f"**Priority:** {result['priority']}")
                                                st.markdown("</div>", unsafe_allow_html=True)

                                # Update progress
                                progress_bar.progress((i + 1) / len(image_paths))

                            except Exception as e:
                                st.error(f"Error processing {os.path.basename(path)}: {str(e)}")

                        status_text.text("✅ Batch processing completed!")

                        # Store batch results in session state
                        st.session_state.batch_results = batch_results
                        reset_pdf_state()  # Reset PDF state for new batch analysis

        # Display stored batch results if available
        if st.session_state.batch_results:
            batch_results = st.session_state.batch_results
            st.header("📊 Batch Analysis Summary")

            # Summary statistics
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)

            with col_stat1:
                st.metric("📁 Total Processed", len(batch_results))

            with col_stat2:
                avg_efficiency = sum(r['predicted_efficiency'] for r in batch_results) / len(batch_results)
                st.metric("📊 Avg Efficiency", f"{avg_efficiency:.1f}%")

            with col_stat3:
                high_priority = sum(1 for r in batch_results if r['priority'] == 'High')
                st.metric("🚨 High Priority", f"{high_priority}/{len(batch_results)}")

            with col_stat4:
                avg_confidence = sum(r['confidence'] for r in batch_results) / len(batch_results)
                st.metric("🎯 Avg Confidence", f"{avg_confidence * 100:.1f}%")

            # Create summary table
            summary_data = []
            for result in batch_results:
                summary_data.append({
                    'Image': result['image_name'],
                    'Defect Type': result['predicted_class'].replace('_', ' ').title(),
                    'Confidence': f"{result['confidence'] * 100:.1f}%",
                    'Efficiency': f"{result['predicted_efficiency']:.1f}%",
                    'Priority': result['priority']
                })

            df = pd.DataFrame(summary_data)
            st.dataframe(df, use_container_width=True)

            # Export options
            st.subheader("💾 Export Results")

            col_export1, col_export2 = st.columns(2)

            with col_export1:
                # CSV Export
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="📄 Download CSV Data",
                    data=csv_data,
                    file_name=f"batch_analysis_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

            with col_export2:
                # PDF Export for batch results
                st.subheader("📄 Generate Batch PDF Report")

                col_pdf_batch1, col_pdf_batch2 = st.columns([1, 1])

                with col_pdf_batch1:
                    if st.button("📄 Generate Batch PDF", key="generate_batch_pdf", type="secondary"):
                        with st.spinner("Generating comprehensive batch PDF report..."):
                            # Try simple batch PDF generation first
                            success = generate_pdf_data_simple(batch_results, is_batch=True)
                            if success:
                                st.success("✅ Batch PDF ready for download!")
                            else:
                                st.error("❌ Failed to generate batch PDF")

                with col_pdf_batch2:
                    # Show download button if PDF is ready for batch
                    if st.session_state.pdf_ready and st.session_state.pdf_data:
                        st.download_button(
                            label="📥 Download Batch PDF",
                            data=st.session_state.pdf_data,
                            file_name=st.session_state.pdf_filename,
                            mime="application/pdf",
                            key="download_batch_pdf",
                            help="Download comprehensive batch analysis report as PDF"
                        )
                        st.success("✅ Batch PDF ready for download!")


if __name__ == "__main__":
    main()