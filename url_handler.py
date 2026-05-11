import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse


def scrape_url(url: str) -> dict:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
    }

    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # ── Title ─────────────────────────────────────────────
    title = ""
    if soup.find("h1"):
        title = soup.find("h1").get_text(strip=True)
    elif soup.title:
        title = soup.title.get_text(strip=True)

    # ── Text ──────────────────────────────────────────────
    # Remove nav, footer, script, style tags first
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    paragraphs = [
        p.get_text(strip=True)
        for p in soup.find_all("p")
        if len(p.get_text(strip=True)) > 40   # skip short/nav paragraphs
    ]
    text = " ".join(paragraphs)

    if not text or len(text) < 100:
        # Fallback — grab all visible text
        text = soup.get_text(separator=" ", strip=True)

    # Limit to 10K words
    text = " ".join(text.split()[:10000])

    # ── Images ────────────────────────────────────────────
    imgs = []
    for img in soup.find_all("img"):
        if len(imgs) >= 5:
            break
        src = img.get("src") or img.get("data-src") or ""
        if not src or src.startswith("data:"):
            continue

        # Resolve relative URLs
        src = urljoin(url, src)
        if not src.startswith("http"):
            continue

        # Skip tiny icons
        try:
            w = int(img.get("width", 0))
            h = int(img.get("height", 0))
            if w and h and (w < 100 or h < 100):
                continue
        except (ValueError, TypeError):
            pass

        imgs.append(src)

    return {
        "title":  title,
        "text":   text,
        "images": imgs
    }