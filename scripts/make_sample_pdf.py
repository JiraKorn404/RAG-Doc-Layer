"""Build a small PDF with text, a table and a figure, for trying out ingestion.

Run: uv run python scripts/make_sample_pdf.py [output.pdf]
"""

import io
import sys
from pathlib import Path

from PIL import Image, ImageDraw
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image as PdfImage
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

REVENUE = [("2021", 12.4), ("2022", 15.1), ("2023", 19.8), ("2024", 26.3)]


def bar_chart() -> io.BytesIO:
    """A bar chart of yearly revenue, drawn by hand so no plotting library is needed."""
    width, height, margin = 640, 400, 60
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.line([(margin, margin), (margin, height - margin)], fill="black", width=3)
    draw.line([(margin, height - margin), (width - margin, height - margin)], fill="black", width=3)
    top = max(value for _, value in REVENUE)
    slot = (width - 2 * margin) / len(REVENUE)
    for index, (year, value) in enumerate(REVENUE):
        left = margin + index * slot + slot * 0.2
        bar_height = (height - 2 * margin) * value / top
        draw.rectangle(
            [left, height - margin - bar_height, left + slot * 0.6, height - margin],
            fill=(40, 90, 200),
        )
        draw.text((left + slot * 0.15, height - margin + 10), year, fill="black")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def build_sample_pdf(path: Path) -> Path:
    styles = getSampleStyleSheet()
    body = styles["BodyText"]
    story = [
        Paragraph("Northwind Robotics Annual Report 2024", styles["Title"]),
        Paragraph("Company overview", styles["Heading2"]),
        Paragraph(
            "Northwind Robotics designs warehouse automation systems. The company was founded "
            "in 2015 in Rotterdam and employs 480 people across three offices. Its flagship "
            "product, the Heron picking arm, was released in March 2023 and is now installed in "
            "more than 200 distribution centres.",
            body,
        ),
        Paragraph(
            "In 2024 the company opened a second assembly plant in Gdansk, which doubled yearly "
            "production capacity to 9,000 units. The chief executive is Mira Okafor.",
            body,
        ),
        Paragraph("Financial results", styles["Heading2"]),
        Paragraph(
            "Revenue grew for the fourth year in a row. The table below shows revenue and "
            "operating profit in millions of euros.",
            body,
        ),
        Spacer(1, 0.4 * cm),
    ]

    rows = [["Year", "Revenue (EUR m)", "Operating profit (EUR m)", "Employees"]]
    profit = {"2021": 1.1, "2022": 1.9, "2023": 3.2, "2024": 5.4}
    staff = {"2021": 210, "2022": 290, "2023": 390, "2024": 480}
    for year, revenue in REVENUE:
        rows.append([year, f"{revenue:.1f}", f"{profit[year]:.1f}", str(staff[year])])
    table = Table(rows, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.75, colors.black),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
            ]
        )
    )
    story += [table, PageBreak()]

    story += [
        Paragraph("Revenue trend", styles["Heading2"]),
        Paragraph("The chart below shows yearly revenue from 2021 to 2024.", body),
        Spacer(1, 0.4 * cm),
        PdfImage(bar_chart(), width=12 * cm, height=7.5 * cm),
        Paragraph("Figure 1: Yearly revenue in millions of euros, 2021 to 2024.", body),
        Paragraph("Outlook", styles["Heading2"]),
        Paragraph(
            "For 2025 the company expects revenue between 31 and 34 million euros and plans to "
            "launch the Heron 2, which lifts packages of up to 35 kilograms.",
            body,
        ),
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(str(path), pagesize=A4, title="Northwind Robotics Annual Report").build(story)
    return path


def build_multipage_pdf(path: Path, pages: int = 10) -> Path:
    """A longer PDF, one section per page, for trying page batches, progress and cancelling.

    Every page has a marker word (`zebra<page>`); every third page has a table and every
    fourth a chart.
    """
    styles = getSampleStyleSheet()
    story = []
    for page in range(1, pages + 1):
        story += [
            Paragraph(f"Section {page}: quarterly notes", styles["Heading2"]),
            Paragraph(
                f"This is page {page} of the logbook. Unit {page} shipped {page * 111} picking "
                f"arms this quarter and its marker word is zebra{page}. " * 4,
                styles["BodyText"],
            ),
        ]
        if page % 3 == 0:
            table = Table(
                [
                    ["Item", "Value"],
                    [f"alpha{page}", str(page * 10)],
                    [f"beta{page}", str(page * 20)],
                ],
                hAlign="LEFT",
            )
            table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.75, colors.black)]))
            story += [Spacer(1, 0.4 * cm), table]
        if page % 4 == 0:
            story += [Spacer(1, 0.4 * cm), PdfImage(bar_chart(), width=12 * cm, height=7.5 * cm)]
        story.append(PageBreak())

    path.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(str(path), pagesize=A4, title=f"Logbook ({pages} pages)").build(story)
    return path


if __name__ == "__main__":
    # make_sample_pdf.py [output.pdf] [pages]   - with a page count, builds the longer logbook
    if len(sys.argv) > 2:
        print(f"Wrote {build_multipage_pdf(Path(sys.argv[1]), int(sys.argv[2]))}")
    else:
        output = (
            Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/samples/northwind_report.pdf")
        )
        print(f"Wrote {build_sample_pdf(output)}")
