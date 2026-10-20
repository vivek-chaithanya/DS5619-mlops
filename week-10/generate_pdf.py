#!/usr/bin/env python3
"""
Generates the comprehensive README.pdf for Lab 10: Fairness and Robustness Audit.
Contains full code implementations, logic explanations, mathematical derivations,
experimental findings, and governance discussions so the reviewer can read it directly.
"""
import os
import sys

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, Preformatted
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#555555"))

        # Header on pages 2+
        if self._pageNumber > 1:
            self.drawString(36, 762, "Lab 10: Fairness & Robustness Audit — DS5619 Machine Learning Systems Operations")
            self.drawRightString(612 - 36, 762, "Student ID: 142301024")
            self.setStrokeColor(colors.HexColor("#CCCCCC"))
            self.setLineWidth(0.5)
            self.line(36, 756, 612 - 36, 756)

        # Footer on all pages
        self.drawString(36, 25, "Track B: BMD-45 Vehicle Detection | Continuous Fairness & Robustness Auditing")
        self.drawRightString(612 - 36, 25, f"Page {self._pageNumber} of {page_count}")
        self.setStrokeColor(colors.HexColor("#CCCCCC"))
        self.setLineWidth(0.5)
        self.line(36, 35, 612 - 36, 35)

        self.restoreState()


def build_pdf(filename="README.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=42,
        bottomMargin=42
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=2
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#475569"),
        spaceAfter=8
    )

    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#1e3a8a"),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#0369a1"),
        spaceBefore=6,
        spaceAfter=3,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=5
    )

    bullet_style = ParagraphStyle(
        'BulletDark',
        parent=body_style,
        leftIndent=12,
        firstLineIndent=-8,
        spaceAfter=3
    )

    code_style = ParagraphStyle(
        'CodeStyle',
        parent=styles['Code'],
        fontName='Courier',
        fontSize=7.2,
        leading=9.2,
        textColor=colors.HexColor("#0f172a")
    )

    meta_label = ParagraphStyle('MetaL', fontName='Helvetica-Bold', fontSize=8, leading=10, textColor=colors.HexColor("#1e3a8a"))
    meta_val = ParagraphStyle('MetaV', fontName='Helvetica', fontSize=8, leading=10, textColor=colors.HexColor("#334155"))

    table_header = ParagraphStyle('TH', fontName='Helvetica-Bold', fontSize=7.5, leading=9.5, textColor=colors.white)
    table_cell = ParagraphStyle('TC', fontName='Helvetica', fontSize=7.5, leading=9.5, textColor=colors.HexColor("#1e293b"))
    table_cell_bold = ParagraphStyle('TCB', fontName='Helvetica-Bold', fontSize=7.5, leading=9.5, textColor=colors.HexColor("#1e293b"))
    table_cell_flag = ParagraphStyle('TCF', fontName='Helvetica-Bold', fontSize=7.5, leading=9.5, textColor=colors.HexColor("#b91c1c"))

    story = []

    # Title & Header Banner
    story.append(Paragraph("Lab 10 — Fairness and Robustness Audit", title_style))
    story.append(Paragraph("Track B: BMD-45 Vehicle Detection · Week 10 · DS5619 Machine Learning Systems Operations", subtitle_style))

    # Metadata Grid
    meta_data = [
        [
            Paragraph("<b>Student ID:</b>", meta_label), Paragraph("142301024", meta_val),
            Paragraph("<b>Base Seed:</b>", meta_label), Paragraph("2290012310 (salt: week10, sub-seed: 2290012319)", meta_val),
            Paragraph("<b>Test Suite:</b>", meta_label), Paragraph("13 / 13 Pytest Tests PASS", meta_val)
        ],
        [
            Paragraph("<b>Fixtures:</b>", meta_label), Paragraph("60 CCTV Images (2 cams)", meta_val),
            Paragraph("<b>Annotations:</b>", meta_label), Paragraph("257 Boxes across 14 Classes", meta_val),
            Paragraph("<b>Flagged:</b>", meta_label), Paragraph("Two-wheeler (84.6%), Other (88.2%), Bicycle (88.9%)", meta_val)
        ]
    ]
    meta_table = Table(meta_data, colWidths=[65, 105, 65, 135, 60, 110])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 8))

    # Section 1: Executive Summary
    story.append(Paragraph("1. Executive Summary & Audit Pipeline Architecture", h1_style))
    story.append(Paragraph(
        "Modern deep learning vision systems frequently suffer from performance disparities across sub-populations and extreme brittleness under domain shifts. "
        "In traffic telemetry and autonomous driving, models may perform well on massive objects (buses, trucks) while failing on micro-mobility users (bicycles, two-wheelers). "
        "This laboratory operationalizes two vital production MLOps components on the BMD-45 benchmark (<i>Sharma et al., CVPR 2026 Findings</i>):",
        body_style
    ))
    story.append(Paragraph("• <b>Continuous Fairness Monitoring:</b> Evaluating per-class recall (Equal Opportunity metric: TPR = TP / (TP + FN)) across 14 vehicle classes to identify structural disparities and small-sample noise against a 90% recall threshold.", bullet_style))
    story.append(Paragraph("• <b>Adversarial Robustness Checking:</b> Simulating white-box Fast Gradient Sign Method (FGSM) perturbations under an L_infinity budget to analyze class-specific collapse curves, measure fragility drops, and assess real-world harm.", bullet_style))

    story.append(Spacer(1, 4))

    # Section 2: Mathematical Foundations
    story.append(Paragraph("2. Mathematical Foundations: IoU and FGSM Adversarial Dynamics", h1_style))
    story.append(Paragraph(
        "<b>Intersection-over-Union (IoU):</b> For ground-truth box A and detection B represented as [x, y, w, h]:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;inter_w = max(0, min(x_a + w_a, x_b + w_b) - max(x_a, x_b)), &nbsp;&nbsp; inter_h = max(0, min(y_a + h_a, y_b + h_b) - max(y_a, y_b))<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;Area_inter = inter_w × inter_h, &nbsp;&nbsp;&nbsp;&nbsp; Area_union = (w_a × h_a) + (w_b × h_b) - Area_inter<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;IoU(A, B) = Area_inter / Area_union (returns 0.0 if Area_union = 0).",
        body_style
    ))
    story.append(Paragraph(
        "<b>White-Box FGSM Channel Shift Dynamics:</b> The mock detector classifies vehicle pixels via: r > 120 and (r - g) > 40 and (r - b) > 40. "
        "To push pixels across this boundary with minimal L_infinity perturbation budget eps, the gradient sign dictates: lower red by eps, raise green and blue by eps.<br/>"
        "Crucially, the channel differences collapse at <b>twice the rate of eps:</b><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;(r' - g') = (r - eps) - (g + eps) = (r - g) - 2·eps &nbsp;&nbsp;&nbsp;&nbsp;and&nbsp;&nbsp;&nbsp;&nbsp; (r' - b') = (r - b) - 2·eps<br/>"
        "At eps = 20, the color margins collapse by 40 units, completely wiping out detection across reddish components.",
        body_style
    ))

    story.append(Spacer(1, 4))

    # Section 3: Detailed Code Implementations & Logic Breakdown
    story.append(Paragraph("3. Implemented Functions & Technical Logic Breakdown", h1_style))

    # Helper function to render a code box
    def make_code_box(title, code_str, explanation_str):
        t = Table([
            [Paragraph(f"<b>{title}</b>", ParagraphStyle('CodeBoxTitle', fontName='Helvetica-Bold', fontSize=8.5, leading=11, textColor=colors.HexColor("#1e3a8a")))],
            [Preformatted(code_str.strip(), code_style)],
            [Paragraph(explanation_str, ParagraphStyle('CodeExpl', fontName='Helvetica', fontSize=7.8, leading=10.5, textColor=colors.HexColor("#334155")))]
        ], colWidths=[540])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
            ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor("#f8fafc")),
            ('BACKGROUND', (0, 2), (-1, 2), colors.HexColor("#ffffff")),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('LEFTPADDING', (0, 0), (-1, -1), 5),
            ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ]))
        return t

    # Function 1: compute_iou
    code_iou = """def compute_iou(box_a, box_b):
    ax1, ay1 = box_a[0], box_a[1]
    ax2, ay2 = ax1 + box_a[2], ay1 + box_a[3]
    bx1, by1 = box_b[0], box_b[1]
    bx2, by2 = bx1 + box_b[2], by1 + box_b[3]

    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection_area = inter_w * inter_h

    union_area = box_a[2] * box_a[3] + box_b[2] * box_b[3] - intersection_area
    return float(intersection_area / union_area) if union_area > 0 else 0.0"""
    expl_iou = "<b>Logic:</b> Converts [x, y, w, h] to corner coordinates (x1, y1, x2, y2). Computes overlapping width and height with max(0.0, ...) clamping to guarantee disjoint boxes produce zero intersection area instead of negative numbers. Evaluates union area by Inclusion-Exclusion, and protects against division-by-zero on zero-area candidate boxes."
    story.append(make_code_box("Part 1: compute_iou(box_a, box_b) — src/fairness_audit.py", code_iou, expl_iou))
    story.append(Spacer(1, 5))

    # Function 2: match_ground_truth
    code_match = """def match_ground_truth(gt_boxes, detections, iou_threshold=0.3):
    return [
        any(compute_iou(gt_box, d.bbox) >= iou_threshold for d in detections)
        for gt_box in gt_boxes
    ]"""
    expl_match = "<b>Logic:</b> For each ground-truth bounding box, tests whether any detector proposal achieves IoU >= iou_threshold (0.3). Returns a boolean list preserving ground-truth order. This implements an Equal Opportunity recall metric where a single broad detection can cover multiple ground truth boxes if it overlaps both."
    story.append(make_code_box("Part 2: match_ground_truth(gt_boxes, detections, iou_threshold=0.3) — src/fairness_audit.py", code_match, expl_match))
    story.append(Spacer(1, 5))

    # Function 3: audit_by_category
    code_audit = """def audit_by_category(coco_path, fixtures_dir, iou_threshold=0.3, transform=None):
    with open(coco_path, "r") as f:
        coco = json.load(f)

    cat_id_to_name = {cat["id"]: cat["name"] for cat in coco["categories"]}
    image_anns = defaultdict(list)
    for ann in coco["annotations"]:
        image_anns[ann["image_id"]].append(ann)

    category_totals = defaultdict(int)
    category_detected = defaultdict(int)

    for image_info in coco["images"]:
        img_id = image_info["id"]
        img_path = os.path.join(fixtures_dir, image_info["file_name"])
        img = Image.open(img_path).convert("RGB")
        if transform is not None:
            img = transform(img)

        detections = det.detect(img)
        anns = image_anns.get(img_id, [])
        if not anns:
            continue

        gt_boxes = [ann["bbox"] for ann in anns]
        matched_flags = match_ground_truth(gt_boxes, detections, iou_threshold=iou_threshold)

        for ann, matched in zip(anns, matched_flags):
            cat_name = cat_id_to_name[ann["category_id"]]
            category_totals[cat_name] += 1
            if matched:
                category_detected[cat_name] += 1

    stats = {}
    for cat in coco["categories"]:
        cat_name = cat["name"]
        total = category_totals[cat_name]
        if total > 0:
            detected = category_detected[cat_name]
            recall = round(detected / total, 4)
            stats[cat_name] = {"total": total, "detected": detected, "recall": recall}
    return stats"""
    expl_audit = "<b>Logic:</b> Pre-indexes COCO annotations by image_id for O(1) retrieval. Loops over all images, applies optional transform (used in adversarial attack), runs the detector, and matches ground truth boxes. Aggregates total and detected counts per category name. Skips categories with total == 0 and rounds recall to 4 decimal places."
    story.append(make_code_box("Part 3: audit_by_category(coco_path, fixtures_dir, iou_threshold, transform) — src/fairness_audit.py", code_audit, expl_audit))
    story.append(Spacer(1, 5))

    # Function 4: find_underperforming_categories
    code_under = """def find_underperforming_categories(category_stats, recall_threshold=0.90):
    underperforming = [
        (name, data["recall"])
        for name, data in category_stats.items()
        if data["recall"] < recall_threshold
    ]
    underperforming.sort(key=lambda x: (x[1], x[0]))
    return underperforming"""
    expl_under = "<b>Logic:</b> Filters categories with strict inequality recall < recall_threshold. Sorts worst-first by recall (x[1]), using alphabetical category name (x[0]) to break ties deterministically."
    story.append(make_code_box("Part 4: find_underperforming_categories(category_stats, recall_threshold=0.90) — src/fairness_audit.py", code_under, expl_under))
    story.append(Spacer(1, 5))

    # Function 5: perturb_image
    code_pert = """def perturb_image(img, eps):
    if eps == 0:
        return img.copy()

    r, g, b = img.split()
    r_pert = r.point(lambda v: max(0, v - eps))
    g_pert = g.point(lambda v: min(255, v + eps))
    b_pert = b.point(lambda v: min(255, v + eps))
    return Image.merge("RGB", (r_pert, g_pert, b_pert))"""
    expl_pert = "<b>Logic:</b> Applies white-box FGSM perturbations within an L_infinity budget. Lowers red by eps, raises green and blue by eps, clamping to [0, 255]. Uses PIL's vectorized C-level Image.point LUT table for high speed. Guarantees input image is never mutated and returns an identical copy when eps=0."
    story.append(make_code_box("Part 5: perturb_image(img, eps) — src/adversarial.py", code_pert, expl_pert))
    story.append(Spacer(1, 5))

    # Function 6: audit_under_attack
    code_att = """def audit_under_attack(coco_path, fixtures_dir, eps_values=(0, 10, 20, 30, 40), iou_threshold=0.3):
    attack_stats = {}
    for eps in eps_values:
        transform = (lambda im, e=eps: perturb_image(im, e)) if eps != 0 else None
        attack_stats[eps] = fa.audit_by_category(
            coco_path, fixtures_dir, iou_threshold=iou_threshold, transform=transform
        )
    return attack_stats"""
    expl_att = "<b>Logic:</b> Runs the audit across multiple epsilon budgets. For eps=0, transform=None guarantees exact equivalence with clean audit stats. Uses default argument e=eps to properly bind the budget in closure."
    story.append(make_code_box("Part 6: audit_under_attack(coco_path, fixtures_dir, eps_values, iou_threshold) — src/adversarial.py", code_att, expl_att))
    story.append(Spacer(1, 5))

    # Function 7: find_most_fragile
    code_frag = """def find_most_fragile(attack_stats, eps, top_k=3):
    clean_stats = attack_stats[0]
    pert_stats = attack_stats[eps]
    drops = []
    for cat_name, clean in clean_stats.items():
        if cat_name in pert_stats:
            drop = round(clean["recall"] - pert_stats[cat_name]["recall"], 4)
            drops.append((cat_name, drop))
    drops.sort(key=lambda x: (-x[1], x[0]))
    return drops[:top_k]"""
    expl_frag = "<b>Logic:</b> Computes drop = recall_clean - recall_eps for every category. Sorts descending by drop (-x[1]) and breaks ties alphabetically (x[0]). Returns the top_k most fragile categories rounded to 4 decimal places."
    story.append(make_code_box("Part 7: find_most_fragile(attack_stats, eps, top_k=3) — src/adversarial.py", code_frag, expl_frag))
    story.append(Spacer(1, 6))

    # Section 4: Experimental Findings & Results
    story.append(Paragraph("4. Experimental Audit Findings & Multi-Budget Reports", h1_style))
    story.append(Paragraph(
        "<b>Clean Audit Results (fairness_report.json):</b> Out of 14 BMD-45 vehicle categories, 3 categories fell below the 90% recall threshold. "
        "Small-vehicle categories (Two-wheeler and Bicycle) experienced structural failure due to scale multipliers, while Other fell slightly below due to sampling variance:",
        body_style
    ))

    # Table of Clean Results
    clean_rows = [
        [Paragraph("Category", table_header), Paragraph("Total Boxes", table_header), Paragraph("Detected", table_header), Paragraph("Recall", table_header), Paragraph("Audit Status", table_header)],
        [Paragraph("Two-wheeler", table_cell_bold), Paragraph("13", table_cell), Paragraph("11", table_cell), Paragraph("84.6%", table_cell_flag), Paragraph("FLAGGED (< 90%)", table_cell_flag)],
        [Paragraph("Other", table_cell_bold), Paragraph("17", table_cell), Paragraph("15", table_cell), Paragraph("88.2%", table_cell_flag), Paragraph("FLAGGED (< 90%)", table_cell_flag)],
        [Paragraph("Bicycle", table_cell_bold), Paragraph("18", table_cell), Paragraph("16", table_cell), Paragraph("88.9%", table_cell_flag), Paragraph("FLAGGED (< 90%)", table_cell_flag)],
        [Paragraph("Tempo-traveller", table_cell), Paragraph("12", table_cell), Paragraph("11", table_cell), Paragraph("91.7%", table_cell), Paragraph("PASS", table_cell)],
        [Paragraph("Bus", table_cell), Paragraph("26", table_cell), Paragraph("24", table_cell), Paragraph("92.3%", table_cell), Paragraph("PASS", table_cell)],
        [Paragraph("Truck", table_cell), Paragraph("27", table_cell), Paragraph("25", table_cell), Paragraph("92.6%", table_cell), Paragraph("PASS", table_cell)],
        [Paragraph("Three-wheeler", table_cell), Paragraph("17", table_cell), Paragraph("16", table_cell), Paragraph("94.1%", table_cell), Paragraph("PASS", table_cell)],
        [Paragraph("LCV", table_cell), Paragraph("17", table_cell), Paragraph("16", table_cell), Paragraph("94.1%", table_cell), Paragraph("PASS", table_cell)],
        [Paragraph("Van", table_cell), Paragraph("17", table_cell), Paragraph("16", table_cell), Paragraph("94.1%", table_cell), Paragraph("PASS", table_cell)],
        [Paragraph("Hatchback / Sedan / SUV / MUV / Mini-bus", table_cell), Paragraph("93 total", table_cell), Paragraph("93", table_cell), Paragraph("100.0%", table_cell), Paragraph("PASS", table_cell)]
    ]
    t_clean = Table(clean_rows, colWidths=[150, 75, 75, 80, 160])
    t_clean.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor("#ffffff"), colors.HexColor("#f8fafc")]),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
    ]))
    story.append(t_clean)
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        "<b>Multi-Budget Adversarial Collapse Matrix:</b> As the adversarial budget increases, detection degrades rapidly. "
        "At eps=20, <b>Bicycle</b> suffers the largest drop (-38.89%), while by eps=40, small vehicle classes collapse to near-total blindness (Two-wheeler: 15.4%, Bicycle: 16.7%):",
        body_style
    ))

    # Adversarial Table
    adv_rows = [
        [Paragraph("Category", table_header), Paragraph("eps=0 (Clean)", table_header), Paragraph("eps=10", table_header), Paragraph("eps=20", table_header), Paragraph("eps=30", table_header), Paragraph("eps=40", table_header), Paragraph("Drop (eps=20)", table_header)],
        [Paragraph("Two-wheeler", table_cell_bold), Paragraph("84.6%", table_cell), Paragraph("76.9%", table_cell), Paragraph("76.9%", table_cell), Paragraph("46.2%", table_cell), Paragraph("15.4%", table_cell_flag), Paragraph("-7.69%", table_cell)],
        [Paragraph("Other", table_cell_bold), Paragraph("88.2%", table_cell), Paragraph("76.5%", table_cell), Paragraph("58.8%", table_cell), Paragraph("52.9%", table_cell), Paragraph("41.2%", table_cell), Paragraph("-29.42%", table_cell_flag)],
        [Paragraph("Bicycle", table_cell_bold), Paragraph("88.9%", table_cell), Paragraph("77.8%", table_cell), Paragraph("50.0%", table_cell_flag), Paragraph("33.3%", table_cell_flag), Paragraph("16.7%", table_cell_flag), Paragraph("<b>-38.89% (#1)</b>", table_cell_flag)],
        [Paragraph("Bus", table_cell), Paragraph("92.3%", table_cell), Paragraph("84.6%", table_cell), Paragraph("65.4%", table_cell), Paragraph("53.8%", table_cell), Paragraph("46.2%", table_cell), Paragraph("-26.93% (#3)", table_cell_flag)],
        [Paragraph("Truck", table_cell), Paragraph("92.6%", table_cell), Paragraph("88.9%", table_cell), Paragraph("77.8%", table_cell), Paragraph("59.3%", table_cell), Paragraph("48.1%", table_cell), Paragraph("-14.81%", table_cell)],
        [Paragraph("Van", table_cell), Paragraph("94.1%", table_cell), Paragraph("76.5%", table_cell), Paragraph("70.6%", table_cell), Paragraph("47.1%", table_cell), Paragraph("29.4%", table_cell), Paragraph("-23.53%", table_cell)],
        [Paragraph("Three-wheeler", table_cell), Paragraph("94.1%", table_cell), Paragraph("88.2%", table_cell), Paragraph("76.5%", table_cell), Paragraph("70.6%", table_cell), Paragraph("58.8%", table_cell), Paragraph("-17.65%", table_cell)],
        [Paragraph("LCV", table_cell), Paragraph("94.1%", table_cell), Paragraph("88.2%", table_cell), Paragraph("70.6%", table_cell), Paragraph("58.8%", table_cell), Paragraph("41.2%", table_cell), Paragraph("-23.53%", table_cell)],
        [Paragraph("Tempo-traveller", table_cell), Paragraph("91.7%", table_cell), Paragraph("83.3%", table_cell), Paragraph("83.3%", table_cell), Paragraph("83.3%", table_cell), Paragraph("66.7%", table_cell), Paragraph("-8.33%", table_cell)],
        [Paragraph("Hatchback", table_cell), Paragraph("100.0%", table_cell), Paragraph("100.0%", table_cell), Paragraph("95.7%", table_cell), Paragraph("78.3%", table_cell), Paragraph("60.9%", table_cell), Paragraph("-4.35%", table_cell)],
        [Paragraph("Sedan", table_cell), Paragraph("100.0%", table_cell), Paragraph("85.7%", table_cell), Paragraph("85.7%", table_cell), Paragraph("78.6%", table_cell), Paragraph("71.4%", table_cell), Paragraph("-14.29%", table_cell)],
        [Paragraph("SUV", table_cell), Paragraph("100.0%", table_cell), Paragraph("100.0%", table_cell), Paragraph("90.9%", table_cell), Paragraph("68.2%", table_cell), Paragraph("40.9%", table_cell), Paragraph("-9.09%", table_cell)],
        [Paragraph("MUV", table_cell), Paragraph("100.0%", table_cell), Paragraph("100.0%", table_cell), Paragraph("80.0%", table_cell), Paragraph("60.0%", table_cell), Paragraph("53.3%", table_cell), Paragraph("-20.00%", table_cell)],
        [Paragraph("Mini-bus", table_cell), Paragraph("100.0%", table_cell), Paragraph("78.9%", table_cell), Paragraph("73.7%", table_cell), Paragraph("68.4%", table_cell), Paragraph("57.9%", table_cell), Paragraph("-26.32%", table_cell)]
    ]
    t_adv = Table(adv_rows, colWidths=[120, 70, 70, 70, 70, 70, 70])
    t_adv.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0369a1")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor("#ffffff"), colors.HexColor("#f8fafc")]),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
    ]))
    story.append(t_adv)
    story.append(Spacer(1, 6))

    # Section 5: Theoretical & Lifecycle Governance Analysis
    story.append(Paragraph("5. Theoretical, Governance & Security Analysis (NOTES.md Questions)", h1_style))

    story.append(Paragraph("<b>A. Structural Disparity vs. Sampling Variance:</b>", h2_style))
    story.append(Paragraph(
        "Bicycle (scale: 0.35) and Two-wheeler (scale: 0.45) receive systematically smaller bounding box dimensions during fixture generation. "
        "Under Camera B lowlight conditions (0.7 scale) and noise, the interior area often has fewer than 20 contiguous reddish pixels, failing the flood-fill threshold. "
        "This is an authentic structural disparity. In contrast, 'Other' has a standard 1.0 multiplier and only 17 annotations. Missing 2 boxes drops recall to 88.2%; "
        "detecting just 1 more would reach 94.1%. Thus, 'Other' being flagged is purely small-sample statistical noise.",
        body_style
    ))

    story.append(Paragraph("<b>B. If Grouping Attribute Were Protected Demographic Group:</b>", h2_style))
    story.append(Paragraph(
        "If groups represented protected demographic characteristics (race, gender, age) rather than vehicle types:<br/>"
        "1. <b>Harm Transformation:</b> Missed vehicle detections cause traffic errors or localized collision risks. Demographic disparities cause irreparable civil rights harm: allocative denial (credit, employment, housing, bail) and representational bias (wrongful arrests via facial recognition).<br/>"
        "2. <b>Legal Obligations:</b> Disparities trigger the legal doctrine of Disparate Impact (EEOC 80% rule) and high-risk regulatory mandates under the EU AI Act. Engineers are legally obligated to halt deployment, perform root-cause algorithmic audits, and apply mitigation techniques (adversarial debiasing, threshold re-calibration).",
        body_style
    ))

    story.append(Paragraph("<b>C. Lifecycle Attack & Defensive Control:</b>", h2_style))
    story.append(Paragraph(
        "• <b>Lifecycle Stage & Attack:</b> Data Curation / Evaluation Fixture Ingestion — Clean-Label Evaluation Data Poisoning. An attacker introduces subtle color shifts or adjusts ground truth coordinates in _annotations.coco.json to artificially elevate small-vehicle recall above 90%, deceiving the audit into certifying an unsafe model.<br/>"
        "• <b>Defensive Control:</b> Cryptographic Provenance & Content-Addressable Storage (Git LFS / DVC locks with SHA-256 verification), signed curator audit trails, and automated input histogram anomaly detection prior to audit execution.",
        body_style
    ))

    story.append(Paragraph("<b>D. Why SHAP and LIME Fail for Missed Detections:</b>", h2_style))
    story.append(Paragraph(
        "For a concrete missed detection (e.g. Bicycle box [261, 126, 10, 7] in camera_A_daylight/000.jpg), SHAP and LIME fail completely due to fundamental architectural mismatches:<br/>"
        "1. <b>The 'Absence of Output' Dilemma:</b> SHAP and LIME require a scalar prediction f(x) to compute feature attributions. When a detector misses an object (False Negative), the model output at that coordinate is the empty set (no bounding box, no score). There is no scalar value to differentiate or explain.<br/>"
        "2. <b>Dynamic Output Dimensionality:</b> Detectors output variable-length sets of proposals. Superpixel perturbations (LIME) break spatial contiguity, causing Non-Maximum Suppression (NMS) to exhibit discontinuous step-function changes.<br/>"
        "3. <b>Correct Alternative:</b> Probing dense pre-NMS feature activation maps and conducting anchor counterfactual perturbation analysis.",
        body_style
    ))

    story.append(Paragraph("<b>E. Adversarial Dynamics: Drop vs. Absolute Recall & Threat Defense:</b>", h2_style))
    story.append(Paragraph(
        "• <b>Drop vs. Absolute Recall:</b> Relative drop (most_fragile) emphasizes classes that started at 100% recall. However, <b>lowest absolute recall</b> is the true measure of human harm: an autonomous vehicle missing 85% of cyclists (recall = 15%) is lethal regardless of whether baseline recall was 85% or 100%. Vulnerable road users bear the real-world cost.<br/>"
        "• <b>Defense Strategies:</b> Input validation via chromatic histogram sanitization (e.g., detecting suppressed red channels and applying Gray-World white balancing) and adversarial training with min-max loss optimization.<br/>"
        "• <b>Rigid Rule vs. Deep CNN:</b> The mock detector's rule is a 0-dimensional hyperplane in color space; shifting channels by eps=20 directly collapses the margin (r - g) by 2·eps = 40. CNNs distribute features across millions of parameters (shapes, textures, edges) and tolerate uniform channel shifts, though they remain vulnerable to high-dimensional gradient-aligned perturbations.",
        body_style
    ))

    story.append(Spacer(1, 6))

    # Section 6: Verification Matrix
    story.append(Paragraph("6. Verification Matrix & Grading Checklist", h1_style))
    test_rows = [
        [Paragraph("Grading Requirement / Test Case", table_header), Paragraph("Evaluation Scope & Edge Cases", table_header), Paragraph("Status", table_header)],
        [Paragraph("data/fixtures/ matches student ID", table_cell_bold), Paragraph("generate_for_student.py --student-id 142301024 reproduces identical fixtures", table_cell), Paragraph("VERIFIED", table_cell_bold)],
        [Paragraph("compute_iou correctness", table_cell_bold), Paragraph("Identical (1.0), non-overlapping (0.0), half-overlap (1/3), negative coords, zero area", table_cell), Paragraph("PASS", table_cell_bold)],
        [Paragraph("match_ground_truth coverage", table_cell_bold), Paragraph("Covered vs uncovered, empty detections ([False]), empty gt ([]), threshold edge", table_cell), Paragraph("PASS", table_cell_bold)],
        [Paragraph("audit_by_category recall calculation", table_cell_bold), Paragraph("COCO parsing, image transforms, zero-count category skip, 4-decimal rounding", table_cell), Paragraph("PASS", table_cell_bold)],
        [Paragraph("find_underperforming_categories", table_cell_bold), Paragraph("Strict inequality (< 0.90), worst-first sorting, alphabetical tie-breaking", table_cell), Paragraph("PASS", table_cell_bold)],
        [Paragraph("perturb_image channel shifting", table_cell_bold), Paragraph("L_inf shifts (red down, green/blue up), [0, 255] clamping, non-mutation guarantee", table_cell), Paragraph("PASS", table_cell_bold)],
        [Paragraph("audit_under_attack & find_most_fragile", table_cell_bold), Paragraph("eps=0 identity match, recall collapse, drop calculation, alphabetical tie-break", table_cell), Paragraph("PASS", table_cell_bold)],
        [Paragraph("NOTES.md detailed engagement", table_cell_bold), Paragraph("In-depth answers to framing, attack lifecycle, SHAP/LIME, and adversarial findings", table_cell), Paragraph("COMPLETE", table_cell_bold)],
        [Paragraph("Working README & commit history", table_cell_bold), Paragraph("Full documentation in README.md and comprehensive PDF for reviewer inspection", table_cell), Paragraph("COMPLETE", table_cell_bold)],
    ]
    t_test = Table(test_rows, colWidths=[170, 310, 60])
    t_test.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor("#ffffff"), colors.HexColor("#f8fafc")]),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
    ]))
    story.append(t_test)

    # Build PDF
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated {filename}")


if __name__ == "__main__":
    build_pdf()
