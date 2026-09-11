import io
import httpx
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from pypdf import PdfReader
from playwright.sync_api import sync_playwright

def _extract_pdf_text(content: bytes) -> str:
    """Extract PDF text and preserve page boundaries as Markdown headings."""
    reader = PdfReader(io.BytesIO(content))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(f"## Page {page_number}\n\n{text}")

    return "\n\n".join(pages)

def scrape_and_tag(
    url: str,
    manufacturer: str,
    model: str,
    model_year: str,
) -> dict:
    is_pdf = url.lower().split("?", 1)[0].endswith(".pdf")

    if is_pdf:
        print(f"   -> Fetching static PDF: {url}")
        response = httpx.get(
            url,
            timeout=30.0,
            follow_redirects=True,
            headers={"User-Agent": "VersionAwareRAG/1.0"},
        )
        response.raise_for_status()
        markdown_text = _extract_pdf_text(response.content)
        source_type = "pdf"
    
    else:
        print(f"   -> Rendering dynamic HTML via Playwright: {url}")
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="networkidle", timeout=60000)
            html_content = page.content()
            browser.close()

        soup = BeautifulSoup(html_content, "html.parser")
        
        # --- Aggressively clean the HTML to prevent LLM parsing failures ---
        # Destroy scripts, styles, SVGs, and hidden elements
        for element in soup(["script", "style", "svg", "path", "symbol", "nav", "footer", "noscript", "iframe"]):
            element.decompose()
        
        main_content = soup.find("main") or soup.find("article") or soup.body
        
        if main_content is None:
            raise ValueError(f"Could not find main document content at {url}")

        # Strip out leftover anchor tags and images to keep the Markdown perfectly clean
        markdown_text = md(str(main_content), heading_style="ATX", strip=["a", "img"])
        source_type = "html"

    metadata = {
        "manufacturer": manufacturer,
        "model": model,
        "model_year": str(model_year),
        "source_url": url,
        "source_type": source_type,
    }

    return {
        "raw_markdown": markdown_text,
        "metadata": metadata,
    }