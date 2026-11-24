"""Print the report / synopsis HTML to PDF with Edge and fill in the contents page numbers.

    python docs/build_pdf.py            # both documents
    python docs/build_pdf.py Synopsis   # one of them

Two passes: print, find on which page each heading/caption landed, write those numbers
into the contents lists, print again. Needs Microsoft Edge (or Chrome) and pypdf.
"""
import re
import subprocess
import sys
from pathlib import Path

from pypdf import PdfReader

DOCS = Path(__file__).resolve().parent
BROWSERS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe"]
ENTRY = re.compile(r'(<li[^>]*><span class="t">)(.*?)(</span><span class="dots"></span><span class="pg">)(\d*)(</span>)')


def render(html, pdf):
    exe = next(b for b in BROWSERS if Path(b).exists())
    subprocess.run([exe, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", html.as_uri()], check=True, capture_output=True)


def norm(s):
    s = re.sub(r"<[^>]+>", "", s).replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    return re.sub(r"\s+", "", s).lower()


def page_of(title, pages, start):
    """First page >= start where a line starts with the title (captions: 'Figure 3.')."""
    m = re.match(r"(Figure|Table) \d+\.", title)
    key = norm(m.group(0) if m else title)[:30]
    for i in range(start, len(pages)):
        if any(norm(line).startswith(key) for line in pages[i]):
            return i
    raise SystemExit(f"not found in PDF: {title!r}")


def fix_toc(html, pdf):
    text = html.read_text(encoding="utf-8")
    pages = [(p.extract_text() or "").splitlines() for p in PdfReader(pdf).pages]
    first = norm(ENTRY.search(text).group(2))
    # the body starts where the first entry stands alone on a line (in the contents it carries a page number)
    body = next(i for i, p in enumerate(pages) if first in map(norm, p))
    cursor = {}  # headings, figures and tables each appear in page order

    def sub(m):
        title = m.group(2)
        k = title.split()[0] if re.match(r"(Figure|Table) \d+\.", title) else "t"
        cursor[k] = page_of(title, pages, cursor.get(k, body))
        return f"{m.group(1)}{title}{m.group(3)}{cursor[k] + 1}{m.group(5)}"

    new = ENTRY.sub(sub, text)
    html.write_text(new, encoding="utf-8")
    return new != text


def build(name):
    html, pdf = DOCS / f"PulseCam_{name}.html", DOCS / f"PulseCam_{name}.pdf"
    render(html, pdf)
    if fix_toc(html, pdf):
        render(html, pdf)
    print(f"{pdf.name}: {len(PdfReader(pdf).pages)} pages")


if __name__ == "__main__":
    for name in sys.argv[1:] or ["Synopsis", "Project_Report"]:
        build(name)
