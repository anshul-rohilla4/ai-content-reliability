from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from datetime import datetime


def generate_pdf(data: dict, filename: str = "truthlens_report.pdf") -> str:
    doc  = SimpleDocTemplate(
        filename,
        pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm,  bottomMargin=2*cm
    )
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=22,
        textColor=colors.HexColor('#101828'),
        spaceAfter=6
    )
    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.HexColor('#667085'),
        spaceAfter=20
    )
    section_style = ParagraphStyle(
        'Section',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#667085'),
        fontName='Helvetica-Bold',
        spaceBefore=16,
        spaceAfter=8,
        textTransform='uppercase',
        letterSpacing=1
    )
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.HexColor('#344054'),
        spaceAfter=6,
        leading=16
    )
    score_style = ParagraphStyle(
        'Score',
        parent=styles['Normal'],
        fontSize=32,
        fontName='Helvetica-Bold',
        textColor=colors.HexColor('#101828'),
        spaceAfter=4
    )

    elements = []

    # ── Header ────────────────────────────────────────────
    elements.append(Paragraph("TruthLens", title_style))
    elements.append(Paragraph("Content Reliability Report", subtitle_style))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e2e5eb')))
    elements.append(Spacer(1, 16))

    # ── Meta info ─────────────────────────────────────────
    meta = [
        ['Analysis Type', data.get('type', 'Content')],
        ['Generated',     datetime.now().strftime('%B %d, %Y at %H:%M')],
    ]
    meta_table = Table(meta, colWidths=[4*cm, 12*cm])
    meta_table.setStyle(TableStyle([
        ('FONTNAME',  (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTSIZE',  (0,0), (-1,-1), 10),
        ('TEXTCOLOR', (0,0), (0,-1), colors.HexColor('#667085')),
        ('TEXTCOLOR', (1,0), (1,-1), colors.HexColor('#344054')),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    elements.append(meta_table)
    elements.append(Spacer(1, 20))

    # ── Reliability Score ─────────────────────────────────
    elements.append(Paragraph("RELIABILITY SCORE", section_style))

    score     = data.get('reliability_score', 0)
    tier      = data.get('tier', '')
    ai_prob   = data.get('ai_probability', '')
    conf      = data.get('confidence', '')

    # Score color based on tier
    tier_lower = tier.lower()
    if 'high' in tier_lower:     score_color = '#027a48'
    elif 'moderate' in tier_lower: score_color = '#b54708'
    elif 'very low' in tier_lower: score_color = '#d92d20'
    else:                          score_color = '#c4320a'

    score_para = ParagraphStyle(
        'ScorePara', parent=styles['Normal'],
        fontSize=36, fontName='Helvetica-Bold',
        textColor=colors.HexColor(score_color)
    )
    elements.append(Paragraph(f"{score} / 100", score_para))

    tier_para = ParagraphStyle(
        'TierPara', parent=styles['Normal'],
        fontSize=14, fontName='Helvetica-Bold',
        textColor=colors.HexColor(score_color),
        spaceAfter=8
    )
    elements.append(Paragraph(tier, tier_para))

    # Score metrics table
    score_data = [
        ['AI Probability', ai_prob],
        ['Confidence',     conf],
        ['Models Agree',   data.get('models_agree', '')],
    ]
    score_table = Table(score_data, colWidths=[5*cm, 11*cm])
    score_table.setStyle(TableStyle([
        ('FONTNAME',  (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTSIZE',  (0,0), (-1,-1), 10),
        ('TEXTCOLOR', (0,0), (0,-1), colors.HexColor('#667085')),
        ('TEXTCOLOR', (1,0), (1,-1), colors.HexColor('#344054')),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8f9fb')),
        ('ROUNDEDCORNERS', [4]),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e5eb')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e5eb')),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('TOPPADDING', (0,0), (-1,-1), 7),
    ]))
    elements.append(score_table)
    elements.append(Spacer(1, 8))

    # ── Input Preview ──────────────────────────────────────
    preview = data.get('input_preview', '')
    if preview:
        elements.append(Paragraph("INPUT PREVIEW", section_style))
        preview_style = ParagraphStyle(
            'Preview', parent=styles['Normal'],
            fontSize=10, textColor=colors.HexColor('#667085'),
            leading=15, borderPadding=(8,8,8,8),
            backColor=colors.HexColor('#f8f9fb'),
            borderColor=colors.HexColor('#e2e5eb'),
            borderWidth=1, borderRadius=4,
            spaceAfter=8
        )
        elements.append(Paragraph(preview, preview_style))

    # ── Contributing Signals ───────────────────────────────
    signals = data.get('signals', {})
    if signals:
        elements.append(Paragraph("CONTRIBUTING SIGNALS", section_style))
        sig_rows = [['Model / Method', 'AI Probability', 'Signal']]
        for model, prob in signals.items():
            prob_val = int(prob.replace('%', '')) if isinstance(prob, str) else int(prob * 100)
            signal   = 'AI Signal' if prob_val >= 50 else 'Human Signal'
            sig_rows.append([model, prob, signal])

        sig_table = Table(sig_rows, colWidths=[7*cm, 4*cm, 5*cm])
        sig_table.setStyle(TableStyle([
            ('BACKGROUND',    (0,0), (-1,0), colors.HexColor('#f1f3f7')),
            ('FONTNAME',      (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE',      (0,0), (-1,-1), 10),
            ('TEXTCOLOR',     (0,0), (-1,0), colors.HexColor('#344054')),
            ('TEXTCOLOR',     (0,1), (-1,-1), colors.HexColor('#667085')),
            ('BOX',           (0,0), (-1,-1), 0.5, colors.HexColor('#e2e5eb')),
            ('INNERGRID',     (0,0), (-1,-1), 0.5, colors.HexColor('#e2e5eb')),
            ('LEFTPADDING',   (0,0), (-1,-1), 10),
            ('TOPPADDING',    (0,0), (-1,-1), 7),
            ('BOTTOMPADDING', (0,0), (-1,-1), 7),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8f9fb')]),
        ]))
        elements.append(sig_table)

    # ── Interpretation ─────────────────────────────────────
    interp = data.get('interpretation', '')
    if interp:
        elements.append(Paragraph("INTERPRETATION", section_style))
        elements.append(Paragraph(interp, body_style))

    # ── Warning ────────────────────────────────────────────
    warning = data.get('warning', '')
    if warning and warning != 'None':
        elements.append(Spacer(1, 8))
        warn_style = ParagraphStyle(
            'Warn', parent=styles['Normal'],
            fontSize=10, textColor=colors.HexColor('#b54708'),
            leading=15, backColor=colors.HexColor('#fffaeb'),
            borderColor=colors.HexColor('#fedf89'),
            borderWidth=1, borderPadding=(8,8,8,8)
        )
        elements.append(Paragraph(f"⚠ {warning}", warn_style))

    # ── Disclaimer ─────────────────────────────────────────
    elements.append(Spacer(1, 24))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e2e5eb')))
    elements.append(Spacer(1, 8))
    disc_style = ParagraphStyle(
        'Disc', parent=styles['Normal'],
        fontSize=9, textColor=colors.HexColor('#98a2b3'),
        leading=14
    )
    elements.append(Paragraph(
        "TruthLens — Content Reliability Scoring System · CSE Final Year Project · "
        "Results are probabilistic and should not be used as the sole basis for punitive action. "
        "For consequential decisions, use a confidence threshold ≥ 0.95 and consult multiple sources.",
        disc_style
    ))

    doc.build(elements)
    return filename