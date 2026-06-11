from __future__ import annotations

import html
import math
import re
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


REPORT_DIR = Path(__file__).resolve().parent
ROOT = REPORT_DIR.parent
ASSET_DIR = REPORT_DIR / "assets"
ASSET_DIR.mkdir(exist_ok=True)

MD_PATH = REPORT_DIR / "Autonomous_Flight_Sentinel_Report.md"
HTML_PATH = REPORT_DIR / "Autonomous_Flight_Sentinel_Report.html"
PDF_PATH = REPORT_DIR / "Autonomous_Flight_Sentinel_Report.pdf"

OUTPUT_DIR = ROOT / "autonomousflight" / "output"
IMAGES = {
    "alarm": OUTPUT_DIR / "ALARM_0004_4_ANNOTATED_20260511_184457.png",
    "bearing": OUTPUT_DIR / "bearing_nav" / "transition_frame_10.png",
    "scan_n": OUTPUT_DIR / "mission" / "scan_N_ep2.png",
    "scan_e": OUTPUT_DIR / "mission" / "scan_E_ep2.png",
    "scan_s": OUTPUT_DIR / "mission" / "scan_S_ep2.png",
    "scan_w": OUTPUT_DIR / "mission" / "scan_W_ep2.png",
    "scan_down": OUTPUT_DIR / "mission" / "scan_DOWN_ep2.png",
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = ["arialbd.ttf" if bold else "arial.ttf", "calibrib.ttf" if bold else "calibri.ttf"]
    for name in names:
        path = Path("C:/Windows/Fonts") / name
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def text_size(text: str, fnt: ImageFont.ImageFont) -> tuple[int, int]:
    tmp = Image.new("RGB", (10, 10))
    box = ImageDraw.Draw(tmp).textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def wrap(text: str, fnt: ImageFont.ImageFont, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if not current or text_size(candidate, fnt)[0] <= width:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


def draw_arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int]) -> None:
    draw.line((*start, *end), fill=(80, 90, 100), width=5)
    ang = math.atan2(end[1] - start[1], end[0] - start[0])
    for delta in (2.55, -2.55):
        x = end[0] + 18 * math.cos(ang + delta)
        y = end[1] + 18 * math.sin(ang + delta)
        draw.line((end[0], end[1], x, y), fill=(80, 90, 100), width=5)


def build_workflow() -> Path:
    out = ASSET_DIR / "workflow.png"
    img = Image.new("RGB", (1600, 620), "white")
    draw = ImageDraw.Draw(img)
    title = font(38, True)
    head = font(24, True)
    body = font(18)
    colors = [(68, 128, 180), (63, 150, 104), (214, 128, 58), (68, 128, 180),
              (63, 150, 104), (214, 128, 58), (180, 70, 65), (23, 50, 77)]
    boxes = [
        (60, 130, 300, 300, "1. Tower", "Fixed AirSim camera captures baseline and monitoring frames."),
        (360, 130, 600, 300, "2. Vision", "Frame difference, HSV smoke/fire mask, exclusion zones, contour area."),
        (660, 130, 900, 300, "3. Bearing", "Pixel centroid is converted to world bearing using FOV and tower yaw."),
        (960, 130, 1200, 300, "4. Phase A", "Drone beam-rides the bearing line with cross-track correction."),
        (1260, 130, 1500, 300, "5. Handoff", "Smoke density above threshold triggers PPO mission phase."),
        (360, 390, 600, 560, "6. Phase C", "Depth image plus target vector drive homing and PPO control."),
        (660, 390, 900, 560, "7. Confirm", "Close-range 360-degree scan confirms fire and saves photos."),
        (960, 390, 1200, 560, "8. Return", "Drone lands and dashboard records mission completion."),
    ]
    draw.text((55, 35), "Autonomous Fire Detection and Response Workflow", font=title, fill=(23, 50, 77))
    for idx, (x1, y1, x2, y2, label, desc) in enumerate(boxes):
        draw.rounded_rectangle((x1, y1, x2, y2), radius=22, fill=colors[idx], outline=(220, 220, 220), width=2)
        draw.text((x1 + 22, y1 + 18), label, font=head, fill="white")
        yy = y1 + 62
        for line in textwrap.wrap(desc, width=24):
            draw.text((x1 + 22, yy), line, font=body, fill="white")
            yy += 24
    for idx in range(4):
        draw_arrow(draw, (boxes[idx][2], 215), (boxes[idx + 1][0], 215))
    draw_arrow(draw, (1380, 300), (480, 390))
    draw_arrow(draw, (600, 475), (660, 475))
    draw_arrow(draw, (900, 475), (960, 475))
    img.save(out)
    return out


def build_scan_sheet() -> Path:
    out = ASSET_DIR / "scan_contact_sheet.png"
    items = [
        ("North scan", IMAGES["scan_n"]),
        ("East scan", IMAGES["scan_e"]),
        ("South scan", IMAGES["scan_s"]),
        ("West scan", IMAGES["scan_w"]),
        ("Downward scan", IMAGES["scan_down"]),
    ]
    thumb_w, thumb_h, pad = 430, 250, 28
    sheet = Image.new("RGB", (pad * 4 + thumb_w * 3, 110 + (thumb_h + 58) * 2 + pad), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((pad, 24), "360-degree confirmation scans", font=font(38, True), fill=(23, 50, 77))
    for idx, (label, path) in enumerate(items):
        row, col = divmod(idx, 3)
        x = pad + col * (thumb_w + pad)
        y = 90 + row * (thumb_h + 58)
        im = Image.open(path).convert("RGB")
        im.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        bg = Image.new("RGB", (thumb_w, thumb_h), (244, 246, 248))
        bg.paste(im, ((thumb_w - im.width) // 2, (thumb_h - im.height) // 2))
        sheet.paste(bg, (x, y))
        draw.rectangle((x, y, x + thumb_w, y + thumb_h), outline=(205, 210, 216), width=2)
        draw.text((x, y + thumb_h + 12), label, font=font(18), fill=(23, 50, 77))
    sheet.save(out)
    return out


def build_system_architecture() -> Path:
    out = ASSET_DIR / "system_architecture.png"
    img = Image.new("RGB", (1500, 820), "white")
    draw = ImageDraw.Draw(img)
    title = font(38, True)
    head = font(22, True)
    body = font(16)
    colors = {
        "tower": (235, 118, 166),
        "vision": (230, 141, 58),
        "control": (69, 128, 182),
        "rl": (74, 153, 107),
        "data": (70, 78, 100),
        "line": (82, 92, 105),
    }

    def box(x: int, y: int, w: int, h: int, color: tuple[int, int, int], label: str, desc: str) -> None:
        draw.rounded_rectangle((x, y, x + w, y + h), radius=18, fill=color, outline=(210, 215, 222), width=2)
        draw.text((x + 18, y + 16), label, font=head, fill="white")
        yy = y + 52
        for line in textwrap.wrap(desc, width=28):
            draw.text((x + 18, yy), line, font=body, fill="white")
            yy += 22

    draw.text((55, 35), "System Architecture", font=title, fill=(23, 50, 77))
    box(70, 130, 270, 150, colors["tower"], "Tower Camera", "FixedCamera1 captures baseline, monitoring, alarm, and annotated images.")
    box(430, 130, 270, 150, colors["vision"], "Vision Module", "OpenCV frame difference, HSV masks, contour filtering, and bearing calculation.")
    box(790, 130, 270, 150, colors["control"], "Mission Controller", "Dispatches drone, arms vehicle, starts Phase A, and handles mission state.")
    box(1150, 130, 270, 150, colors["rl"], "PPO Environment", "Depth image and target vector guide close-range approach and scan.")
    box(250, 430, 270, 150, colors["control"], "Bearing Navigator", "Cross-track correction keeps drone on the tower bearing ray.")
    box(610, 430, 270, 150, colors["data"], "Evidence Storage", "Saves tower photos, transition frames, scan images, logs, and model files.")
    box(970, 430, 270, 150, colors["tower"], "Dashboard", "Displays status, log stream, mission timer, and photo gallery.")

    draw_arrow(draw, (340, 205), (430, 205))
    draw_arrow(draw, (700, 205), (790, 205))
    draw_arrow(draw, (1060, 205), (1150, 205))
    draw_arrow(draw, (925, 280), (415, 430))
    draw_arrow(draw, (520, 505), (610, 505))
    draw_arrow(draw, (880, 505), (970, 505))
    draw_arrow(draw, (1285, 280), (745, 430))
    img.save(out)
    return out


def build_placeholder(path: Path, bg: tuple[int, int, int], title_text: str, subtitle: str) -> Path:
    img = Image.new("RGB", (1400, 760), bg)
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((45, 45, 1355, 715), radius=28, outline=(255, 255, 255), width=6)
    draw.text((95, 100), title_text, font=font(54, True), fill="white")
    y = 190
    for line in textwrap.wrap(subtitle, width=58):
        draw.text((100, y), line, font=font(28), fill=(255, 255, 255))
        y += 42
    draw.text((100, 625), "Replace this PNG with your real screenshot using the same file name.", font=font(24, True), fill="white")
    img.save(path)
    return path


def build_placeholders() -> None:
    build_placeholder(
        ASSET_DIR / "dashboard_placeholder.png",
        (226, 86, 162),
        "DASHBOARD SCREENSHOT PLACEHOLDER",
        "Put the real dashboard image here. Suggested view: mission status, console logs, timer, sector view, and saved mission photos.",
    )
    build_placeholder(
        ASSET_DIR / "drone_flight_placeholder.png",
        (231, 126, 44),
        "DRONE FLIGHT PHOTO PLACEHOLDER",
        "Put the real in-flight drone picture here. Suggested view: drone flying toward the detected smoke or fire source.",
    )


def inline_markup(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", text)
    return text


def table_to_html(rows: list[str]) -> str:
    headers = [c.strip() for c in rows[0].strip("|").split("|")]
    body = [[c.strip() for c in row.strip("|").split("|")] for row in rows[2:]]
    parts = ["<table><thead><tr>"]
    parts.extend(f"<th>{inline_markup(h)}</th>" for h in headers)
    parts.append("</tr></thead><tbody>")
    for row in body:
        parts.append("<tr>")
        parts.extend(f"<td>{inline_markup(c)}</td>" for c in row)
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def build_html(md: str) -> None:
    lines = md.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(lines[i])
                i += 1
            out.append(table_to_html(rows))
            continue
        if line.startswith("# "):
            out.append(f"<h1>{inline_markup(line[2:])}</h1>")
        elif line.startswith("## "):
            out.append(f"<h2>{inline_markup(line[3:])}</h2>")
        elif line.startswith("### "):
            out.append(f"<h3>{inline_markup(line[4:])}</h3>")
        elif line.startswith("!["):
            alt = line[line.find("[") + 1: line.find("]")]
            src = line[line.find("(") + 1: line.rfind(")")]
            out.append(f'<figure><img src="{html.escape(src)}" alt="{html.escape(alt)}"></figure>')
        elif line.strip().startswith(tuple(f"{n}." for n in range(1, 10))):
            out.append(f"<p class='numbered'>{inline_markup(line)}</p>")
        elif line.strip():
            out.append(f"<p>{inline_markup(line)}</p>")
        i += 1

    css = """
body{margin:0;background:#f5f7fb;color:#1f2933;font-family:Arial,Helvetica,sans-serif}
main{max-width:980px;margin:0 auto;background:white;padding:54px 70px 80px;box-shadow:0 16px 50px rgba(15,23,42,.12)}
h1{font-size:36px;line-height:1.12;margin:0 0 24px;color:#17324d}
h2{margin-top:42px;padding-top:18px;border-top:2px solid #d8dee7;color:#17324d;font-size:25px}
h3{margin-top:28px;color:#1d6b7a;font-size:19px}
p{font-size:16px;line-height:1.58} code{background:#eef2f7;padding:1px 4px;border-radius:4px}
table{width:100%;border-collapse:collapse;margin:18px 0 26px;font-size:14px}
th{background:#17324d;color:white;text-align:left} th,td{padding:10px 12px;border:1px solid #d8dee7;vertical-align:top}
tr:nth-child(even) td{background:#f8fafc} figure{margin:28px 0} figure img{width:100%;border:1px solid #d8dee7;border-radius:6px}
@media print{body{background:white}main{box-shadow:none;padding:24px 34px}}
"""
    HTML_PATH.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Autonomous Flight Sentinel Report</title>"
        f"<style>{css}</style></head><body><main>{''.join(out)}</main></body></html>",
        encoding="utf-8",
    )


class Pdf:
    W = 1240
    H = 1754
    M = 86
    INK = (32, 41, 54)
    MUTED = (92, 106, 123)
    ACCENT = (29, 107, 122)
    NAVY = (23, 50, 77)
    RULE = (216, 222, 231)
    LIGHT = (246, 248, 251)

    def __init__(self) -> None:
        self.pages: list[Image.Image] = []
        self.page: Image.Image | None = None
        self.draw: ImageDraw.ImageDraw | None = None
        self.y = self.M
        self.no = 0
        self.cover = font(48, True)
        self.cover_sub = font(24)
        self.h1 = font(30, True)
        self.h2 = font(22, True)
        self.body = font(18)
        self.body_b = font(18, True)
        self.small = font(14)
        self.small_b = font(14, True)
        self.foot = font(12)

    @property
    def body_width(self) -> int:
        return self.W - 2 * self.M

    def start(self) -> None:
        self.no += 1
        self.page = Image.new("RGB", (self.W, self.H), "white")
        self.draw = ImageDraw.Draw(self.page)
        self.y = self.M
        if self.no > 1:
            self.draw.text((self.M, 38), "Autonomous Flight Sentinel Report", font=self.foot, fill=self.MUTED)
            self.draw.line((self.M, 62, self.W - self.M, 62), fill=self.RULE, width=2)
            self.y = 92

    def finish(self) -> None:
        if self.page is None or self.draw is None:
            return
        self.draw.line((self.M, self.H - 62, self.W - self.M, self.H - 62), fill=self.RULE)
        self.draw.text((self.W - self.M - 70, self.H - 48), f"Page {self.no}", font=self.foot, fill=self.MUTED)
        self.pages.append(self.page)
        self.page = None
        self.draw = None

    def ensure(self, height: int) -> None:
        if self.y + height > self.H - 92:
            self.finish()
            self.start()

    def cover_page(self, title: str, md: str) -> None:
        self.start()
        assert self.draw and self.page
        self.draw.rectangle((0, 0, self.W, 210), fill=self.NAVY)
        self.draw.text((self.M, 60), "Autonomous Flight Sentinel", font=self.cover, fill="white")
        self.draw.text((self.M, 122), "Fire Detection and Autonomous Drone Response in AirSim", font=self.cover_sub, fill=(225, 235, 242))
        self.draw.text((self.M, 270), "Project Report", font=self.h1, fill=self.NAVY)
        for idx, line in enumerate(["Prepared by: [Student Name Surname]", "Supervisor: [Dr. Name Surname]", "Date: 11 May 2026"]):
            self.draw.text((self.M, 326 + idx * 30), line, font=self.body, fill=self.INK)
        im = Image.open(IMAGES["alarm"]).convert("RGB")
        im.thumbnail((self.body_width, 680), Image.Resampling.LANCZOS)
        self.page.paste(im, (self.M, 520))
        self.draw.rectangle((self.M, 520, self.M + im.width, 520 + im.height), outline=self.RULE, width=2)
        self.y = 520 + im.height + 28
        self.para("This report follows the supplied template structure and uses actual mission output images and logs from the autonomous flight simulation project.", self.body, self.MUTED, 0)
        self.finish()

    def heading(self, text: str, level: int) -> None:
        fnt = self.h1 if level == 2 else self.h2
        color = self.NAVY if level == 2 else self.ACCENT
        self.ensure(70)
        assert self.draw
        if level == 2:
            self.draw.line((self.M, self.y, self.W - self.M, self.y), fill=self.RULE, width=2)
            self.y += 18
        self.draw.text((self.M, self.y), text, font=fnt, fill=color)
        self.y += 44 if level == 2 else 34

    def para(self, text: str, fnt=None, color=None, space: int = 18) -> None:
        fnt = fnt or self.body
        color = color or self.INK
        lines = wrap(text, fnt, self.body_width)
        self.ensure(len(lines) * 25 + space)
        assert self.draw
        for line in lines:
            self.draw.text((self.M, self.y), line, font=fnt, fill=color)
            self.y += 25
        self.y += space

    def numbered(self, text: str) -> None:
        m = re.match(r"(\d+\.)\s+(.*)", text)
        if not m:
            self.para(text)
            return
        prefix, rest = m.groups()
        lines = wrap(rest, self.body, self.body_width - 42)
        self.ensure(len(lines) * 25 + 8)
        assert self.draw
        self.draw.text((self.M, self.y), prefix, font=self.body, fill=self.INK)
        for line in lines:
            self.draw.text((self.M + 42, self.y), line, font=self.body, fill=self.INK)
            self.y += 25
        self.y += 5

    def image(self, md_line: str) -> None:
        alt = md_line[md_line.find("[") + 1: md_line.find("]")]
        src = md_line[md_line.find("(") + 1: md_line.rfind(")")]
        path = REPORT_DIR / src
        im = Image.open(path).convert("RGB")
        max_h = 590 if im.width > im.height else 520
        scale = min(self.body_width / im.width, max_h / im.height)
        w, h = int(im.width * scale), int(im.height * scale)
        self.ensure(h + 60)
        im = im.resize((w, h), Image.Resampling.LANCZOS)
        x = self.M + (self.body_width - w) // 2
        assert self.page and self.draw
        self.draw.rectangle((x - 2, self.y - 2, x + w + 2, self.y + h + 2), outline=self.RULE, width=2)
        self.page.paste(im, (x, self.y))
        self.y += h + 12
        self.para(alt, self.small, self.MUTED, 10)

    def table(self, rows: list[str]) -> None:
        headers = [c.strip() for c in rows[0].strip("|").split("|")]
        data = [[c.strip() for c in r.strip("|").split("|")] for r in rows[2:]]
        if len(headers) == 3 and headers[0] == "Time":
            widths = [130, 280, self.body_width - 410]
        elif len(headers) == 3:
            widths = [310, 170, self.body_width - 480]
        else:
            widths = [self.body_width // len(headers)] * len(headers)
            widths[-1] += self.body_width - sum(widths)

        blocks = []
        for row in [headers] + data:
            cells = []
            max_lines = 1
            for value, width in zip(row, widths):
                lines = wrap(value.replace("`", ""), self.small_b if row == headers else self.small, width - 18)
                cells.append(lines)
                max_lines = max(max_lines, len(lines))
            blocks.append((cells, max(34, max_lines * 20 + 16)))
        self.ensure(sum(h for _, h in blocks) + 20)
        assert self.draw
        yy = self.y
        for row_index, (cells, height) in enumerate(blocks):
            xx = self.M
            for idx, lines in enumerate(cells):
                fill = self.NAVY if row_index == 0 else (self.LIGHT if row_index % 2 == 0 else "white")
                self.draw.rectangle((xx, yy, xx + widths[idx], yy + height), fill=fill, outline=self.RULE)
                color = "white" if row_index == 0 else self.INK
                fnt = self.small_b if row_index == 0 else self.small
                ty = yy + 8
                for line in lines:
                    self.draw.text((xx + 9, ty), line, font=fnt, fill=color)
                    ty += 20
                xx += widths[idx]
            yy += height
        self.y = yy + 22

    def save(self) -> None:
        if self.page is not None:
            self.finish()
        self.pages[0].save(PDF_PATH, save_all=True, append_images=self.pages[1:], resolution=150.0, quality=85)


def strip_markup(text: str) -> str:
    return text.replace("**", "").replace("*", "").replace("`", "")


def build_pdf(md: str) -> None:
    pdf = Pdf()
    title = md.splitlines()[0].lstrip("# ")
    pdf.cover_page(title, md)
    pdf.start()
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if i < 7:
            i += 1
            continue
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(lines[i])
                i += 1
            pdf.table(rows)
            continue
        if line.startswith("## "):
            pdf.heading(strip_markup(line[3:]), 2)
        elif line.startswith("### "):
            pdf.heading(strip_markup(line[4:]), 3)
        elif line.startswith("!["):
            pdf.image(line)
        elif line.startswith("**Figure"):
            pdf.para(strip_markup(line), pdf.small, pdf.MUTED, 14)
        elif line.strip().startswith(tuple(f"{n}." for n in range(1, 10))):
            pdf.numbered(strip_markup(line))
        elif line.strip():
            pdf.para(strip_markup(line), pdf.body_b if line.startswith("`") else pdf.body)
        i += 1
    pdf.save()


def main() -> None:
    build_workflow()
    build_scan_sheet()
    build_system_architecture()
    build_placeholders()
    md = MD_PATH.read_text(encoding="utf-8")
    build_html(md)
    build_pdf(md)
    print(PDF_PATH)
    print(HTML_PATH)
    print(MD_PATH)


if __name__ == "__main__":
    main()
