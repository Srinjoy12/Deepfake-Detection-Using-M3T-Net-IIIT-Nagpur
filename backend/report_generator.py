from fpdf import FPDF
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime
import os

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Deepfake Analysis Report', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_report(analysis_result: dict) -> bytes:
    pdf = PDF()
    pdf.add_page()
    
    # Analysis Summary
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(0, 10, 'Analysis Summary', 0, 1, 'L')
    pdf.set_font('helvetica', '', 10)
    
    current_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    sanitized_filename = os.path.basename(analysis_result.get('filename', 'N/A'))

    duration = analysis_result.get('video_duration_seconds')
    frames = analysis_result.get('total_frames')
    windows = analysis_result.get('windows_analyzed')

    summary_data = {
        "Report Generated": current_time,
        "Analyzed File": sanitized_filename,
        "Verdict": "Deepfake Detected" if analysis_result.get('is_deepfake') else "Authentic",
        "Confidence": f"{analysis_result.get('confidence', 0) * 100:.2f}%",
        "Video Duration": f"{duration:.2f} seconds" if duration is not None else "N/A",
        "Total Frames": str(frames) if frames is not None else "N/A",
        "Windows Analyzed": str(windows) if windows is not None else "N/A",
    }

    for key, value in summary_data.items():
        pdf.set_font('helvetica', 'B', 10)
        pdf.cell(40, 6, f"{key}:", 0, 0, 'L')
        pdf.set_font('helvetica', '', 10)
        pdf.cell(0, 6, value, 0, 1, 'L')

    pdf.ln(10)

    # Evidence Section
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(0, 10, 'Evidence', 0, 1, 'L')
    pdf.set_font('helvetica', 'B', 10)
    
    face_images_b64 = analysis_result.get('face_images_b64', [])
    
    if not face_images_b64:
        pdf.cell(0, 10, 'No face crops available.', 0, 1, 'L')
    else:
        pdf.cell(0, 10, 'Key Frame Face Crops:', 0, 1, 'L')
        pdf.ln(2)

        # Calculate image layout
        num_images = len(face_images_b64)
        img_width = 50 
        img_height = 50
        margin = 10
        max_width = pdf.w - 2 * pdf.l_margin
        images_per_row = int(max_width / (img_width + margin))
        if images_per_row == 0: images_per_row = 1
        
        x_start = pdf.get_x()
        y_start = pdf.get_y()
        x = x_start
        y = y_start

        for i, face_b64 in enumerate(face_images_b64):
            try:
                img_bytes = base64.b64decode(face_b64)
                img = Image.open(io.BytesIO(img_bytes))
                
                # Check if we need to move to the next row
                if i > 0 and i % images_per_row == 0:
                    y += img_height + margin
                    x = x_start

                pdf.image(img, x=x, y=y, w=img_width, h=img_height)
                x += img_width + margin

            except Exception as e:
                print(f"Error processing image for PDF: {e}")
                continue
        
        # Move Y position down past the images
        num_rows = (num_images + images_per_row - 1) // images_per_row
        pdf.set_y(y_start + num_rows * (img_height + margin))


    pdf.ln(10)

    # Per-Window Probability Analysis
    pdf.set_font('helvetica', 'B', 10)
    pdf.cell(0, 10, 'Per-Window Probability Analysis:', 0, 1)
    
    probs = analysis_result.get('probabilities', [])
    if probs:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(probs, marker='o', linestyle='-', color='b')
        ax.axhline(y=0.5, color='r', linestyle='--', label='Deepfake Threshold (>0.5)')
        ax.axhline(y=0.03, color='g', linestyle='--', label='Authentic Threshold (<0.03)')
        ax.set_title('Deepfake Probability per Video Window')
        ax.set_xlabel('Window Index')
        ax.set_ylabel('Probability')
        ax.legend()
        ax.grid(True)
        
        plot_buffer = io.BytesIO()
        plt.savefig(plot_buffer, format='png', bbox_inches='tight')
        plt.close(fig)
        plot_buffer.seek(0)
        
        pdf.image(plot_buffer, w=pdf.w - 40)
    else:
        pdf.set_font('Arial', 'I', 10)
        pdf.cell(0, 10, 'No probability data available.', 0, 1)

    return pdf.output(dest='S') 