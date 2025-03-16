import argparse
import json
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

def generate_report(input_file, output_file):
    # Load analysis results
    with open(input_file, 'r') as f:
        results = json.load(f)
    
    # Create PDF report
    doc = SimpleDocTemplate(output_file, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    
    # Title
    title_style = styles["Title"]
    elements.append(Paragraph("Candidate Interview Analysis Report", title_style))
    elements.append(Spacer(1, 20))
    
    # Overall Score
    if "evaluation_results" in results and "Overall_Score" in results["evaluation_results"]:
        score = results["evaluation_results"]["Overall_Score"]
        heading_style = styles["Heading2"]
        elements.append(Paragraph("Overall Score", heading_style))
        elements.append(Spacer(1, 10))
        
        score_style = ParagraphStyle(
            'Score',
            parent=styles["Normal"],
            fontSize=16,
            alignment=1
        )
        elements.append(Paragraph(f"<b>{score}/10</b>", score_style))
        elements.append(Spacer(1, 20))
    
    # Gaze Analysis
    if "gaze_report" in results:
        gaze = results["gaze_report"]
        elements.append(Paragraph("Gaze Analysis", styles["Heading2"]))
        elements.append(Spacer(1, 10))
        
        gaze_data = [
            ["Suspicion Score", f"{gaze.get('suspicion_score', 'N/A')}/100"],
            ["Suspicion Level", gaze.get('suspicion_level', 'N/A')],
            ["Behavior Assessment", gaze.get('behavior_assessment', 'N/A')],
            ["Reading Behavior", f"{gaze.get('reading_percentage', 0):.1f}% of interview"]
        ]
        
        gaze_table = Table(gaze_data, colWidths=[200, 300])
        gaze_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(gaze_table)
        elements.append(Spacer(1, 20))
    
    # Detailed Feedback
    if "evaluation_results" in results and "Feedback" in results["evaluation_results"]:
        feedback = results["evaluation_results"]["Feedback"]
        elements.append(Paragraph("Detailed Feedback", styles["Heading2"]))
        elements.append(Spacer(1, 10))
        
        # Summary
        if "Summary" in feedback and feedback["Summary"]:
            elements.append(Paragraph("Summary", styles["Heading3"]))
            for point in feedback["Summary"]:
                elements.append(Paragraph(f"• {point}", styles["Normal"]))
            elements.append(Spacer(1, 10))
        
        # Strengths
        if "Strengths" in feedback and feedback["Strengths"]:
            elements.append(Paragraph("Strengths", styles["Heading3"]))
            for strength in feedback["Strengths"]:
                elements.append(Paragraph(f"• {strength}", styles["Normal"]))
            elements.append(Spacer(1, 10))
        
        # Areas for Improvement
        if "Areas_for_Improvement" in feedback and feedback["Areas_for_Improvement"]:
            elements.append(Paragraph("Areas for Improvement", styles["Heading3"]))
            for area in feedback["Areas_for_Improvement"]:
                elements.append(Paragraph(f"• {area}", styles["Normal"]))
            elements.append(Spacer(1, 10))
    
    # Interview Transcript (abbreviated)
    if "segments" in results and results["segments"]:
        elements.append(Paragraph("Interview Transcript", styles["Heading2"]))
        elements.append(Spacer(1, 10))
        
        # Just include the first few exchanges
        max_segments = min(10, len(results["segments"]))
        for i in range(max_segments):
            segment = results["segments"][i]
            speaker = segment["speaker"]
            text = segment["text"]
            
            speaker_style = ParagraphStyle(
                'Speaker',
                parent=styles["Normal"],
                fontName='Helvetica-Bold'
            )
            elements.append(Paragraph(speaker, speaker_style))
            elements.append(Paragraph(text, styles["Normal"]))
            elements.append(Spacer(1, 5))
        
        if len(results["segments"]) > max_segments:
            elements.append(Paragraph("(Transcript abbreviated for brevity)", styles["Italic"]))
    
    # Build PDF document
    doc.build(elements)
    print(f"Report generated successfully: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate PDF report from analysis results")
    parser.add_argument("--input", required=True, help="Path to analysis results JSON file")
    parser.add_argument("--output", required=True, help="Path to output PDF file")
    
    args = parser.parse_args()
    generate_report(args.input, args.output) 