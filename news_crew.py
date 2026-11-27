"""
news_crew.py - Two-agent CrewAI crew: tech newsletter + newspaper-style PDF.

Researcher (SerperDevTool, news search) -> Task 1: find today's top 3 tech stories
                                           (headline, summary, source, image URL).
Writer (NO tools)                       -> Task 2: turn them into a newsletter of
                                           EXACTLY 5 lines (headline, 3 stories, sign-off).
reportlab                               -> lays the stories + newsletter out as a front page.

The tasks run with Process.sequential, and the Writer's task receives the
Researcher's output through `context=[research_task]`.

Setup:
    pip install crewai crewai-tools reportlab pillow requests python-dotenv
    # .env (or real environment variables):
    #   OPENAI_API_KEY=sk-...
    #   SERPER_API_KEY=...

Run:
    python news_crew.py            # full crew run -> tech_times_YYYY-MM-DD.pdf
    python news_crew.py --demo     # layout preview with sample data (no API keys needed)
    python news_crew.py -o my.pdf  # choose the output file name
"""

import argparse
import html
import io
import os
import re
import sys
import tempfile
from datetime import date
from typing import List, Optional
from urllib.parse import urljoin, urlparse

import requests
from dotenv import load_dotenv
from PIL import Image as PILImage
from pydantic import BaseModel, Field
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

load_dotenv(override=True)

PAPER_NAME = "The Tech Times"
STORY_COUNT = 3
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


# --------------------------------------------------------------------------
# 1. Data shape passed between the agents (and into the PDF)
# --------------------------------------------------------------------------
class Story(BaseModel):
    headline: str = Field(description="Short newspaper headline")
    summary: str = Field(description="2-3 sentence summary of the story")
    source_url: str = Field(description="URL of the original article")
    image_url: Optional[str] = Field(
        default=None, description="Image URL from the search result, or null"
    )


class Edition(BaseModel):
    stories: List[Story]


# --------------------------------------------------------------------------
# 2. The crew
# --------------------------------------------------------------------------
MAX_ATTEMPTS = 3  # re-run the crew if the Researcher returns fewer than 3 stories


def _lines(text: str) -> List[str]:
    """Non-empty lines, with markdown/bullet decoration stripped."""
    out = []
    for ln in (text or "").strip().splitlines():
        ln = re.sub(r"[*#`]", "", ln).strip()
        ln = re.sub(r"^(?:[-\u2022]|\d+[.)])\s+", "", ln)
        if ln:
            out.append(ln)
    return out


def five_line_guardrail(result):
    """Task guardrail: the Writer's output must be exactly 5 non-empty lines.

    If not, CrewAI sends the message below back to the Writer and retries.
    """
    lines = _lines(result.raw)
    if len(lines) == 5:
        return True, "\n".join(lines)
    return False, (
        f"Your output had {len(lines)} lines. It must be EXACTLY 5 lines: "
        "1 headline line, 3 story lines (one per story), 1 sign-off line. "
        "No blank lines, no numbering, no extra text."
    )


def build_crew():
    # Imported here so `--demo` works without crewai installed or keys set.
    from crewai import Agent, Crew, Process, Task
    from crewai_tools import SerperDevTool

    today = date.today().strftime("%B %d, %Y")

    # search_type="news" returns news results, which usually include an imageUrl.
    search_tool = SerperDevTool(search_type="news", n_results=10)

    # ---- Agents: each has its own role, goal and backstory ----
    researcher = Agent(
        role="Tech News Researcher",
        goal=f"Find the {STORY_COUNT} most important tech news stories of today ({today}).",
        backstory=(
            "You monitor the tech industry daily. You verify that a story is "
            "recent, prefer reputable sources, and always keep the source link "
            "and the image link exactly as the search tool returned them."
        ),
        tools=[search_tool],  # the ONLY agent that can search
        allow_delegation=False,
        verbose=True,
    )

    writer = Agent(
        role="Newsletter Writer",
        goal="Turn researched tech stories into a punchy newsletter of exactly 5 lines.",
        backstory=(
            "You are a newsletter editor known for tight, energetic copy. You "
            "cannot browse the web, so you rely only on the research you are "
            "handed and never invent facts, numbers or links."
        ),
        tools=[],  # the Writer has no tools
        allow_delegation=False,
        verbose=True,
    )

    # ---- Task 1: research ----
    research_task = Task(
        description=(
            f"Search the web for the top {STORY_COUNT} tech news stories of today, "
            f"{today}. You MUST return exactly {STORY_COUNT} different stories on "
            "different topics - never fewer. If a search gives fewer than "
            f"{STORY_COUNT} good stories, search again with other keywords (for "
            "example AI, chips, startups, cybersecurity, big tech). If you cannot "
            "find enough stories published today, use the most recent ones from "
            "the last 48 hours. For each story capture the headline, a factual "
            "2-sentence summary, the article URL and the image URL (the "
            "'imageUrl' field of the search result, or null). Never invent URLs."
        ),
        expected_output=(
            f"Exactly {STORY_COUNT} stories, each with: headline, summary, "
            "source_url and image_url (or null)."
        ),
        agent=researcher,
        output_pydantic=Edition,
    )

    # ---- Task 2: write the 5-line newsletter ----
    writing_task = Task(
        description=(
            "Using ONLY the research you were given, write a short tech "
            "newsletter. Do not add facts that are not in the research.\n\n"
            "Format, exactly 5 lines:\n"
            "  Line 1: a catchy newsletter headline.\n"
            "  Lines 2-4: one line per story (one story each, one sentence each).\n"
            "  Line 5: a short sign-off line.\n"
            "No blank lines, no numbering, no text before or after."
        ),
        expected_output=(
            "Exactly 5 lines of plain text: 1 headline line, 3 story lines, "
            "1 sign-off line. The final output must be exactly 5 lines - no blank "
            "lines, no preamble, no extra lines."
        ),
        agent=writer,
        # THE HANDOFF: the Researcher's result does not arrive on its own;
        # `context` injects research_task's output into this task's prompt.
        context=[research_task],
        guardrail=five_line_guardrail,  # enforces "exactly 5 lines"
    )

    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, writing_task],
        process=Process.sequential,
        verbose=True,
    )
    return crew, research_task, writing_task


def _stories_from_task(task) -> List[Story]:
    """Read the Researcher's structured stories, de-duplicated by URL/headline."""
    out = getattr(task, "output", None)
    edition = getattr(out, "pydantic", None)
    if edition is None:
        try:
            raw = re.sub(r"^```(?:json)?|```$", "", (out.raw or "").strip(), flags=re.M)
            edition = Edition.model_validate_json(raw.strip())
        except Exception:
            return []
    seen, stories = set(), []
    for s in edition.stories:
        key = (s.source_url or s.headline).strip().lower()
        if key not in seen:
            seen.add(key)
            stories.append(s)
    return stories


def run_crew():
    """Run the crew; returns (Edition of stories, 5-line newsletter text)."""
    missing = [k for k in ("OPENAI_API_KEY", "SERPER_API_KEY") if not os.environ.get(k)]
    if missing:
        sys.exit(f"Missing environment variable(s): {', '.join(missing)}")

    best_stories, newsletter = [], ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        crew, research_task, writing_task = build_crew()
        try:
            result = crew.kickoff()
        except Exception as exc:  # e.g. the 5-line guardrail kept failing
            print(f"\n[attempt {attempt}] crew failed: {exc}")
            continue

        stories = _stories_from_task(research_task)
        if len(stories) > len(best_stories):
            best_stories, newsletter = stories, str(result).strip()
        if len(best_stories) >= STORY_COUNT:
            break
        print(f"\n[attempt {attempt}] only {len(stories)} stories found, retrying...")

    if not best_stories:
        sys.exit("The crew did not return any stories. Please try again.")
    if len(best_stories) < STORY_COUNT:
        print(f"\n[warning] Only {len(best_stories)} of {STORY_COUNT} stories were found.")
    return Edition(stories=best_stories[:STORY_COUNT]), newsletter


# --------------------------------------------------------------------------
# 3. Images: download, fall back to the article's og:image, crop to fit
# --------------------------------------------------------------------------
def _fetch_bytes(src: str) -> bytes:
    if os.path.exists(src):  # local file (used by --demo)
        with open(src, "rb") as f:
            return f.read()
    r = requests.get(src, headers={"User-Agent": USER_AGENT}, timeout=15)
    r.raise_for_status()
    return r.content


def _find_og_image(page_url: str) -> Optional[str]:
    """Look for <meta property="og:image"> on the article page."""
    try:
        r = requests.get(page_url, headers={"User-Agent": USER_AGENT}, timeout=10)
        page = r.text[:300_000]
    except Exception:
        return None
    patterns = (
        r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)',
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:image["\']',
    )
    for pattern in patterns:
        m = re.search(pattern, page, re.I)
        if m:
            return urljoin(page_url, html.unescape(m.group(1)))
    return None


def _crop_to_aspect(img: PILImage.Image, aspect: float) -> PILImage.Image:
    w, h = img.size
    if w / h > aspect:  # too wide: trim the sides
        new_w = int(h * aspect)
        left = (w - new_w) // 2
        return img.crop((left, 0, left + new_w, h))
    new_h = int(w / aspect)  # too tall: trim the bottom, keep the top
    top = (h - new_h) // 3
    return img.crop((0, top, w, top + new_h))


def get_image_flowable(story: Story, width: float, aspect: float) -> Optional[Image]:
    """Return a reportlab Image sized to `width`, or None if nothing usable."""
    candidates = [story.image_url] if story.image_url else []
    for src in candidates + ["__og__"]:
        try:
            if src == "__og__":
                src = _find_og_image(story.source_url)
                if not src:
                    continue
            img = PILImage.open(io.BytesIO(_fetch_bytes(src))).convert("RGB")
            if img.width < 200 or img.height < 120:  # icons / tracking pixels
                continue
            img = _crop_to_aspect(img, aspect)
            img.thumbnail((1400, 1400))
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=85)
            buf.seek(0)
            return Image(buf, width=width, height=width / aspect)
        except Exception:
            continue
    return None


# --------------------------------------------------------------------------
# 4. Newspaper layout
# --------------------------------------------------------------------------
INK = colors.HexColor("#1a1a1a")
ACCENT = colors.HexColor("#9b1c1c")
LINK = colors.HexColor("#1a4f8b")


def _styles():
    def S(name, **kw):
        kw.setdefault("textColor", INK)
        return ParagraphStyle(name, **kw)

    return {
        "masthead": S("masthead", fontName="Times-Bold", fontSize=40, leading=43, alignment=TA_CENTER),
        "strip": S("strip", fontName="Times-Roman", fontSize=9, leading=11),
        "strip_c": S("strip_c", fontName="Times-Bold", fontSize=9, leading=11, alignment=TA_CENTER),
        "kicker": S("kicker", fontName="Helvetica-Bold", fontSize=8, leading=10, textColor=ACCENT, spaceAfter=3),
        "lead_h": S("lead_h", fontName="Times-Bold", fontSize=26, leading=29, spaceAfter=6),
        "h": S("h", fontName="Times-Bold", fontSize=16, leading=19, spaceAfter=6),
        "lead_body": S("lead_body", fontName="Times-Roman", fontSize=11.5, leading=15.5, alignment=TA_JUSTIFY),
        "body": S("body", fontName="Times-Roman", fontSize=10.5, leading=14, alignment=TA_JUSTIFY),
        "source": S("source", fontName="Times-Italic", fontSize=8.5, leading=11),
        "brief_h": S("brief_h", fontName="Times-Bold", fontSize=13, leading=16, spaceAfter=3),
        "brief": S("brief", fontName="Times-Roman", fontSize=10, leading=13, leftIndent=10, bulletIndent=0, spaceAfter=1.5),
        "brief_end": S("brief_end", fontName="Times-Italic", fontSize=10, leading=13, spaceBefore=2),
    }


def _clean(text: str) -> str:
    """Escape for reportlab markup and drop glyphs the built-in fonts cannot draw."""
    text = (text or "").encode("cp1252", "ignore").decode("cp1252")
    return html.escape(text, quote=False)


def _story_flowables(story: Story, st, width: float, lead: bool, kicker: str, aspect: float):
    items = [
        Paragraph(kicker, st["kicker"]),
        Paragraph(_clean(story.headline), st["lead_h" if lead else "h"]),
    ]
    img = get_image_flowable(story, width, aspect)
    if img:
        items += [img, Spacer(1, 7)]
    items.append(Paragraph(_clean(story.summary), st["lead_body" if lead else "body"]))
    items.append(Spacer(1, 4))
    domain = urlparse(story.source_url).netloc.replace("www.", "") or "source"
    href = html.escape(story.source_url, quote=True)
    items.append(
        Paragraph(f'Source: <link href="{href}" color="#1a4f8b">{_clean(domain)}</link>', st["source"])
    )
    return items


def _briefing_box(newsletter: str, st, width: float):
    """The Writer's exactly-5-line newsletter, shown as a boxed panel."""
    lines = _lines(newsletter)
    if not lines:
        return None
    cell = [Paragraph("THE BRIEFING &nbsp;|&nbsp; NEWSLETTER IN 5 LINES", st["kicker"])]
    cell.append(Paragraph(_clean(lines[0]), st["brief_h"]))
    for ln in lines[1:-1]:
        cell.append(Paragraph(_clean(ln), st["brief"], bulletText="\u2022"))
    if len(lines) > 1:
        cell.append(Paragraph(_clean(lines[-1]), st["brief_end"]))
    box = Table([[cell]], colWidths=[width])
    box.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 1.2, INK),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f4f1ea")),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return box


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(INK)
    canvas.setLineWidth(0.5)
    canvas.line(1.5 * cm, 1.2 * cm, A4[0] - 1.5 * cm, 1.2 * cm)
    canvas.setFont("Times-Italic", 8)
    canvas.drawCentredString(A4[0] / 2, 0.8 * cm, f"{PAPER_NAME}  |  Page {doc.page}  |  Compiled by an AI crew")
    canvas.restoreState()


def build_pdf(edition: Edition, newsletter: str, out_path: str) -> str:
    st = _styles()
    margin = 1.5 * cm
    content_w = A4[0] - 2 * margin
    gutter = 0.35 * cm

    doc = SimpleDocTemplate(
        out_path,
        pagesize=A4,
        leftMargin=margin,
        rightMargin=margin,
        topMargin=1.2 * cm,
        bottomMargin=1.5 * cm,
        title=f"{PAPER_NAME} - {date.today():%d %B %Y}",
        author="CrewAI",
    )

    story = []

    # Masthead
    story.append(Paragraph(PAPER_NAME.upper(), st["masthead"]))
    story.append(HRFlowable(width="100%", thickness=3, color=INK, spaceBefore=2, spaceAfter=1.5))
    story.append(HRFlowable(width="100%", thickness=0.8, color=INK, spaceAfter=3))

    strip = Table(
        [[
            Paragraph("DAILY TECHNOLOGY EDITION", st["strip"]),
            Paragraph(date.today().strftime("%A, %d %B %Y").upper(), st["strip_c"]),
            Paragraph("<para align='right'>ALL THE NEWS THAT'S FIT TO CODE</para>", st["strip"]),
        ]],
        colWidths=[content_w / 3] * 3,
    )
    strip.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(strip)
    story.append(Spacer(1, 10))

    stories = edition.stories
    if not stories:
        story.append(Paragraph("No stories were available today.", st["body"]))
        doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
        return out_path

    # Lead story: full width, wide image
    story += _story_flowables(stories[0], st, content_w, lead=True, kicker="TOP STORY", aspect=3.6)
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=0.8, color=INK, spaceAfter=8))

    # Remaining stories: two columns per row, thin rule between them
    col_w = content_w / 2
    inner_w = col_w - gutter
    rest = stories[1:]
    for i in range(0, len(rest), 2):
        pair = rest[i:i + 2]
        cells = [
            _story_flowables(s, st, inner_w, lead=False, kicker="TECHNOLOGY", aspect=2.4)
            for s in pair
        ]
        while len(cells) < 2:
            cells.append([""])
        row = Table([cells], colWidths=[col_w, col_w])
        row.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (0, 0), gutter),
            ("LEFTPADDING", (1, 0), (1, 0), gutter),
            ("LINEAFTER", (0, 0), (0, 0), 0.6, INK),
        ]))
        story.append(row)
        story.append(Spacer(1, 6))

    # The Writer's 5-line newsletter
    box = _briefing_box(newsletter, st, content_w)
    if box is not None:
        story.append(box)

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return out_path


# --------------------------------------------------------------------------
# 5. Demo data (layout preview without API keys)
# --------------------------------------------------------------------------
def _demo_image(path: str, c1, c2):
    img = PILImage.new("RGB", (1200, 700))
    px = img.load()
    for y in range(700):
        for x in range(1200):
            t = (x / 1200 + y / 700) / 2
            px[x, y] = tuple(int(c1[k] + (c2[k] - c1[k]) * t) for k in range(3))
    img.save(path, "JPEG")


DEMO_NEWSLETTER = (
    "THE TECH TIMES: THREE STORIES THAT MATTERED TODAY\n"
    "Chipmakers unveil cheaper AI accelerators as demand for compute keeps surging.\n"
    "A major open-source browser ships a faster, safer rebuilt rendering engine.\n"
    "A startup reveals a low-cost satellite internet terminal for rural homes.\n"
    "That's your tech briefing - stay curious."
)


def demo_edition() -> Edition:
    tmp = tempfile.mkdtemp()
    imgs = []
    for i, (a, b) in enumerate([((20, 40, 90), (120, 170, 220)),
                                ((90, 30, 30), (230, 150, 90)),
                                ((20, 80, 60), (150, 210, 170))]):
        p = os.path.join(tmp, f"demo{i}.jpg")
        _demo_image(p, a, b)
        imgs.append(p)
    return Edition(stories=[
        Story(
            headline="Chipmakers Race To Build Cheaper AI Hardware As Demand Surges",
            summary=("Leading semiconductor firms announced new accelerator designs aimed at cutting the "
                     "cost of running large AI models. Analysts say the move could ease data-centre "
                     "bottlenecks, though supply constraints remain a concern for buyers this year."),
            source_url="https://www.example.com/chips-ai-hardware",
            image_url=imgs[0],
        ),
        Story(
            headline="Open-Source Browser Update Promises Faster, Safer Browsing",
            summary=("A major open-source browser shipped an update with a rebuilt rendering engine and "
                     "stronger tracking protection. Early benchmarks show noticeable gains in page load times."),
            source_url="https://www.example.org/browser-update",
            image_url=imgs[1],
        ),
        Story(
            headline="Startup Unveils Low-Cost Satellite Internet Terminal",
            summary=("A new entrant in satellite broadband revealed a compact home terminal priced well "
                     "below current rivals. The company plans pilot rollouts in rural regions next quarter, "
                     "pending regulatory approval."),
            source_url="https://www.example.net/satellite-terminal",
            image_url=imgs[2],
        ),
    ])


# --------------------------------------------------------------------------
# 6. Entry point
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Generate a newspaper-style tech news PDF.")
    ap.add_argument("--demo", action="store_true", help="use sample data instead of running the crew")
    ap.add_argument("-o", "--output", help="output PDF path")
    args = ap.parse_args()

    out = args.output or f"tech_times_{date.today():%Y-%m-%d}.pdf"
    if args.demo:
        edition, newsletter = demo_edition(), DEMO_NEWSLETTER
    else:
        edition, newsletter = run_crew()

    print("\n" + "=" * 60)
    print("FINAL NEWSLETTER (5 lines)")
    print("=" * 60)
    print(newsletter)
    line_count = len(_lines(newsletter))
    if line_count != 5:
        print(f"\n[warning] Expected 5 lines but got {line_count}.")

    build_pdf(edition, newsletter, out)
    print(f"\nSaved newspaper: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()