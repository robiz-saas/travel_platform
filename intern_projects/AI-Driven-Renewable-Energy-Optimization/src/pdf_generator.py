# src/pdf_generator.py
"""
PDF Report Generator for Solar Panel Analysis
Creates professional PDF reports with analysis results, charts, and recommendations
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics import renderPDF
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
import io
import base64
import os
import tempfile
import shutil
from PIL import Image as PILImage
import numpy as np
import time


class SolarPanelPDFGenerator:
    """Generate comprehensive PDF reports for solar panel analysis"""

    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()

        # Create dedicated temp directory for this session
        self.temp_dir = tempfile.mkdtemp(prefix="solar_pdf_")
        self.temp_files = []  # Track files for cleanup

        # Ensure matplotlib doesn't interfere
        plt.ioff()  # Turn off interactive mode

    def setup_custom_styles(self):
        """Define custom styles for the PDF"""

        # Title style
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        )

        # Header style
        self.header_style = ParagraphStyle(
            'CustomHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            spaceBefore=20,
            textColor=colors.darkgreen
        )

        # Subheader style
        self.subheader_style = ParagraphStyle(
            'CustomSubHeader',
            parent=self.styles['Heading3'],
            fontSize=14,
            spaceAfter=8,
            spaceBefore=12,
            textColor=colors.darkblue
        )

        # Body text style
        self.body_style = ParagraphStyle(
            'CustomBody',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=6,
            alignment=TA_LEFT
        )

        # Recommendation style
        self.recommendation_style = ParagraphStyle(
            'Recommendation',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=4,
            leftIndent=20,
            bulletIndent=10
        )

    def get_temp_filename(self, suffix=".tmp"):
        """Generate a unique temporary filename in our temp directory"""
        timestamp = int(time.time() * 1000)  # millisecond precision
        pid = os.getpid()
        filename = f"temp_{pid}_{timestamp}{suffix}"
        filepath = os.path.join(self.temp_dir, filename)
        self.temp_files.append(filepath)
        return filepath

    def cleanup_temp_files(self):
        """Clean up all temporary files created during this session"""
        for filepath in self.temp_files:
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except Exception as e:
                print(f"Warning: Could not remove temp file {filepath}: {e}")

        # Clean up temp directory
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            print(f"Warning: Could not remove temp directory {self.temp_dir}: {e}")

    def create_efficiency_gauge_image(self, efficiency_value):
        """Create an efficiency gauge visualization with proper temp file handling"""

        gauge_filename = self.get_temp_filename(".png")

        try:
            fig, ax = plt.subplots(figsize=(6, 4), facecolor='white')

            # Create gauge
            theta = np.linspace(0, np.pi, 100)

            # Gauge background
            ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=8, alpha=0.3)

            # Efficiency arc
            efficiency_theta = np.pi * (efficiency_value / 100)
            efficiency_arc = np.linspace(0, efficiency_theta, 50)

            # Color based on efficiency
            if efficiency_value >= 75:
                color = 'green'
            elif efficiency_value >= 50:
                color = 'orange'
            else:
                color = 'red'

            ax.plot(np.cos(efficiency_arc), np.sin(efficiency_arc), color=color, linewidth=8)

            # Add needle
            needle_angle = efficiency_theta
            ax.arrow(0, 0, 0.8 * np.cos(needle_angle), 0.8 * np.sin(needle_angle),
                     head_width=0.05, head_length=0.1, fc='black', ec='black')

            # Add text
            ax.text(0, -0.3, f'{efficiency_value:.1f}%',
                    ha='center', va='center', fontsize=20, fontweight='bold')
            ax.text(0, -0.5, 'Efficiency',
                    ha='center', va='center', fontsize=14)

            # Formatting
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-0.7, 1.2)
            ax.set_aspect('equal')
            ax.axis('off')

            # Save to temp file
            plt.tight_layout()
            plt.savefig(gauge_filename, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)  # Explicitly close figure

            return gauge_filename

        except Exception as e:
            print(f"Error creating efficiency gauge: {e}")
            plt.close('all')  # Close all figures on error
            return None

    def create_priority_chart(self, priority):
        """Create a priority level visualization with proper temp file handling"""

        priority_filename = self.get_temp_filename(".png")

        try:
            fig, ax = plt.subplots(figsize=(6, 2), facecolor='white')

            # Priority levels and colors
            levels = ['Low', 'Medium', 'High']
            colors_map = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}

            # Create bars
            for i, level in enumerate(levels):
                color = colors_map[level] if level == priority else 'lightgray'
                alpha = 1.0 if level == priority else 0.3
                ax.barh(0, 1, left=i, height=0.5, color=color, alpha=alpha, edgecolor='black')
                ax.text(i + 0.5, 0, level, ha='center', va='center', fontweight='bold')

            # Highlight current priority
            if priority in levels:
                priority_index = levels.index(priority)
                ax.barh(0, 1, left=priority_index, height=0.5,
                        color=colors_map[priority], alpha=1.0, edgecolor='black', linewidth=3)

            ax.set_xlim(0, 3)
            ax.set_ylim(-0.5, 0.5)
            ax.set_title(f'Maintenance Priority: {priority}', fontsize=14, fontweight='bold')
            ax.axis('off')

            plt.tight_layout()
            plt.savefig(priority_filename, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)  # Explicitly close figure

            return priority_filename

        except Exception as e:
            print(f"Error creating priority chart: {e}")
            plt.close('all')  # Close all figures on error
            return None

    def process_analysis_image(self, image_path):
        """Process and resize analysis image for PDF inclusion"""
        if not image_path or not os.path.exists(image_path):
            return None

        try:
            temp_img_path = self.get_temp_filename(".jpg")

            # Open and resize image
            img = PILImage.open(image_path)
            img.thumbnail((400, 300), PILImage.Resampling.LANCZOS)

            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Save to temp file
            img.save(temp_img_path, "JPEG", quality=85)

            return temp_img_path

        except Exception as e:
            print(f"Error processing image: {e}")
            return None

    def generate_single_analysis_report(self, analysis_result, image_path=None, output_path=None):
        """
        Generate PDF report for single image analysis

        Args:
            analysis_result (dict): Analysis results from your AI model
            image_path (str): Path to analyzed image
            output_path (str): Output path for PDF
        """

        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.get_temp_filename(".pdf")

        try:
            # Create PDF document
            doc = SimpleDocTemplate(output_path, pagesize=A4,
                                    rightMargin=72, leftMargin=72,
                                    topMargin=72, bottomMargin=18)

            # Container for PDF elements
            elements = []

            # Title
            title = Paragraph("🔍 Solar Panel Analysis Report", self.title_style)
            elements.append(title)
            elements.append(Spacer(1, 20))

            # Report metadata
            report_info = [
                ['Report Generated:', datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                ['Analysis Type:', 'AI-Powered Defect Detection & Efficiency Prediction'],
                ['Model Type:', analysis_result.get('model_type', 'Multi-Output Neural Network')]
            ]

            report_table = Table(report_info, colWidths=[2 * inch, 4 * inch])
            report_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(report_table)
            elements.append(Spacer(1, 30))

            # Image section (if image provided)
            processed_image_path = self.process_analysis_image(image_path)
            if processed_image_path:
                elements.append(Paragraph("📸 Analyzed Image", self.header_style))
                try:
                    img_element = Image(processed_image_path, width=4 * inch, height=3 * inch)
                    elements.append(img_element)
                    elements.append(Spacer(1, 20))
                except Exception as e:
                    print(f"Could not add image to PDF: {e}")
                    elements.append(Paragraph(f"Could not load image: {str(e)}", self.body_style))

            # Analysis Results Section
            elements.append(Paragraph("🔍 Analysis Results", self.header_style))

            # Main findings table
            if analysis_result.get('success', False):
                findings = [
                    ['Metric', 'Value', 'Status'],
                    ['Defect Type', analysis_result.get('predicted_class', 'Unknown').replace('_', ' ').title(), ''],
                    ['Confidence', f"{analysis_result.get('confidence', 0) * 100:.1f}%",
                     '✅ High' if analysis_result.get('confidence', 0) > 0.8 else '⚠️ Moderate' if analysis_result.get(
                         'confidence', 0) > 0.6 else '❌ Low'],
                    ['Predicted Efficiency', f"{analysis_result.get('predicted_efficiency', 0):.1f}%",
                     '✅ Good' if analysis_result.get('predicted_efficiency',
                                                     0) > 70 else '⚠️ Fair' if analysis_result.get(
                         'predicted_efficiency', 0) > 40 else '❌ Poor'],
                    ['Maintenance Priority', analysis_result.get('priority', 'Unknown'),
                     '🚨 Urgent' if analysis_result.get('priority') == 'High' else '⚠️ Soon' if analysis_result.get(
                         'priority') == 'Medium' else '✅ Routine']
                ]

                findings_table = Table(findings, colWidths=[2 * inch, 1.5 * inch, 1.5 * inch])
                findings_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 0), (-1, -1), 11),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                elements.append(findings_table)
                elements.append(Spacer(1, 30))

                # Try to add efficiency gauge
                efficiency_value = analysis_result.get('predicted_efficiency', 0)
                gauge_filename = self.create_efficiency_gauge_image(efficiency_value)
                if gauge_filename and os.path.exists(gauge_filename):
                    try:
                        gauge_img = Image(gauge_filename, width=3 * inch, height=2 * inch)
                        elements.append(gauge_img)
                        elements.append(Spacer(1, 20))
                    except Exception as e:
                        print(f"Could not add efficiency gauge: {e}")

                # Try to add priority chart
                priority = analysis_result.get('priority', 'Low')
                priority_filename = self.create_priority_chart(priority)
                if priority_filename and os.path.exists(priority_filename):
                    try:
                        priority_img = Image(priority_filename, width=3 * inch, height=1 * inch)
                        elements.append(priority_img)
                        elements.append(Spacer(1, 30))
                    except Exception as e:
                        print(f"Could not add priority chart: {e}")

            # Recommendations Section
            recommendations = analysis_result.get('recommendations', [])
            if recommendations:
                elements.append(Paragraph("💡 Maintenance Recommendations", self.header_style))

                for i, rec in enumerate(recommendations, 1):
                    rec_text = f"{i}. {rec}"
                    elements.append(Paragraph(rec_text, self.recommendation_style))

                elements.append(Spacer(1, 20))

            # Technical Details Section
            elements.append(Paragraph("🔬 Technical Details", self.header_style))

            if 'all_probabilities' in analysis_result:
                elements.append(Paragraph("Classification Probabilities:", self.subheader_style))

                for class_name, prob in analysis_result['all_probabilities'].items():
                    detail_text = f"• {class_name.replace('_', ' ').title()}: {prob * 100:.1f}%"
                    elements.append(Paragraph(detail_text, self.body_style))

            # Efficiency details
            if 'efficiency' in analysis_result and isinstance(analysis_result['efficiency'], dict):
                eff_data = analysis_result['efficiency']
                elements.append(Spacer(1, 10))
                elements.append(Paragraph("Efficiency Analysis Details:", self.subheader_style))

                if 'confidence_interval' in eff_data:
                    ci = eff_data['confidence_interval']
                    detail_text = f"• Confidence Interval: {ci[0]:.1f}% - {ci[1]:.1f}%"
                    elements.append(Paragraph(detail_text, self.body_style))

                if 'description' in eff_data:
                    detail_text = f"• Analysis: {eff_data['description']}"
                    elements.append(Paragraph(detail_text, self.body_style))

            # Disclaimer
            elements.append(Spacer(1, 30))
            disclaimer = """
            <b>Disclaimer:</b> This analysis is generated by an AI model and should be used as a 
            guidance tool. Professional inspection is recommended for critical maintenance decisions. 
            The efficiency predictions are estimates based on visual analysis and may vary from 
            actual performance measurements.
            """
            elements.append(Paragraph(disclaimer, self.body_style))

            # Footer
            elements.append(Spacer(1, 20))
            footer_text = f"Generated by Solar Panel AI Analysis System | Report ID: {datetime.now().strftime('%Y%m%d%H%M%S')}"
            footer_style = ParagraphStyle('Footer', parent=self.styles['Normal'],
                                          fontSize=8, alignment=TA_CENTER, textColor=colors.grey)
            elements.append(Paragraph(footer_text, footer_style))

            # Build PDF
            doc.build(elements)
            print(f"📄 PDF report generated: {output_path}")

            return output_path

        except Exception as e:
            print(f"Error generating single analysis PDF: {e}")
            raise e

    def generate_batch_analysis_report(self, batch_results, output_path=None):
        """
        Generate PDF report for batch analysis

        Args:
            batch_results (list): List of analysis results
            output_path (str): Output path for PDF
        """

        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.get_temp_filename(".pdf")

        try:
            # Create PDF document
            doc = SimpleDocTemplate(output_path, pagesize=A4,
                                    rightMargin=72, leftMargin=72,
                                    topMargin=72, bottomMargin=18)

            elements = []

            # Title
            title = Paragraph("📊 Solar Panel Batch Analysis Report", self.title_style)
            elements.append(title)
            elements.append(Spacer(1, 20))

            # Summary statistics
            if batch_results:
                total_panels = len(batch_results)
                avg_efficiency = np.mean([r.get('predicted_efficiency', 0) for r in batch_results])
                high_priority = sum(1 for r in batch_results if r.get('priority') == 'High')

                summary_data = [
                    ['Summary Metric', 'Value'],
                    ['Total Panels Analyzed', str(total_panels)],
                    ['Average Efficiency', f"{avg_efficiency:.1f}%"],
                    ['High Priority Issues', f"{high_priority} ({high_priority / total_panels * 100:.1f}%)"],
                    ['Report Generated', datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
                ]

                summary_table = Table(summary_data, colWidths=[3 * inch, 2 * inch])
                summary_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.darkgreen),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 11),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                elements.append(summary_table)
                elements.append(Spacer(1, 30))

                # Detailed results table
                elements.append(Paragraph("📋 Detailed Analysis Results", self.header_style))

                # Create table data
                table_data = [['Image', 'Defect Type', 'Confidence', 'Efficiency', 'Priority']]

                for result in batch_results[:20]:  # Limit to first 20 for PDF space
                    table_data.append([
                        result.get('image_name', 'Unknown')[:20] + '...' if len(
                            result.get('image_name', '')) > 20 else result.get('image_name', 'Unknown'),
                        result.get('predicted_class', 'Unknown').replace('_', ' ').title(),
                        f"{result.get('confidence', 0) * 100:.1f}%",
                        f"{result.get('predicted_efficiency', 0):.1f}%",
                        result.get('priority', 'Unknown')
                    ])

                results_table = Table(table_data,
                                      colWidths=[1.2 * inch, 1.2 * inch, 0.8 * inch, 0.8 * inch, 0.8 * inch])
                results_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
                ]))
                elements.append(results_table)

                if len(batch_results) > 20:
                    elements.append(Spacer(1, 10))
                    note = f"Note: Showing first 20 results. Total analyzed: {len(batch_results)} images."
                    elements.append(Paragraph(note, self.body_style))

            # Build PDF
            doc.build(elements)
            print(f"📄 Batch PDF report generated: {output_path}")

            return output_path

        except Exception as e:
            print(f"Error generating batch analysis PDF: {e}")
            raise e

    def __del__(self):
        """Cleanup temp files when object is destroyed"""
        try:
            self.cleanup_temp_files()
        except:
            pass


# Example usage and testing
if __name__ == "__main__":
    # Test PDF generator
    generator = SolarPanelPDFGenerator()

    try:
        # Sample analysis result for testing
        sample_result = {
            'success': True,
            'predicted_class': 'dusty',
            'confidence': 0.87,
            'predicted_efficiency': 35.7,
            'priority': 'High',
            'model_type': 'AI Neural Network',
            'recommendations': [
                'Immediate cleaning required',
                'Increase cleaning frequency',
                'Consider automated cleaning systems'
            ],
            'all_probabilities': {
                'dusty': 0.87,
                'clean': 0.05,
                'bird_droppings': 0.04,
                'electrical_damage': 0.02,
                'physical_damage': 0.01,
                'snow_covered': 0.01
            },
            'efficiency': {
                'confidence_interval': (8.6, 62.8),
                'description': 'Dust significantly reduces light transmission'
            }
        }

        # Generate test PDF
        pdf_path = generator.generate_single_analysis_report(sample_result)
        print(f"✅ Test PDF generated: {pdf_path}")

    finally:
        # Always cleanup
        generator.cleanup_temp_files()