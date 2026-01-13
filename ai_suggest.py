import os
import re
import html
import feedparser
import requests
from datetime import date
from dotenv import load_dotenv
from urllib.parse import quote_plus

# --- CONFIG ---
TOPIC = "microwave plasma CVD diamond growth"
MAX_RESULTS = 10
OUTPUT_DIR = "digests"
# Default model — change to a model available on OpenRouter or set OPENROUTER_MODEL in env
MODEL = os.getenv("OPENROUTER_MODEL", "mistralai/mistral-7b-instruct")
OPENROUTER_URL = os.getenv("OPENROUTER_URL", "https://api.openrouter.ai/v1/chat/completions")
# ---------------

load_dotenv()
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")

USER_AGENT = os.getenv("SCIPAN_USER_AGENT", "SciPAn/1.0 (mailto:you@example.com)")

def search_arxiv(query, max_results=10):
    """Fetch papers from arXiv."""
    q = quote_plus(query)
    url = f"https://export.arxiv.org/api/query?search_query=all:{q}&sortBy=submittedDate&sortOrder=descending&max_results={max_results}"
    headers = {"User-Agent": USER_AGENT}
    try:
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
    except Exception:
        # Return empty list on network failure — caller can handle
        return []

    feed = feedparser.parse(r.text)
    papers = []
    for entry in feed.entries:
        summary = html.unescape(getattr(entry, "summary", "")).strip()
        # remove simple HTML tags
        summary = re.sub(r"<[^>]+>", "", summary)
        authors = ", ".join(a.name for a in getattr(entry, "authors", []))
        papers.append({
            "title": getattr(entry, "title", "").strip(),
            "summary": summary,
            "link": getattr(entry, "link", ""),
            "authors": authors,
            "published": getattr(entry, "published", ""),
            "id": getattr(entry, "id", ""),
        })
    return papers

def summarize_text(text):
    """Summarize text using OpenRouter-hosted models (fallback to short extractive summary).

    Requires `OPENROUTER_API_KEY` in the environment. If the key is missing or the
    call fails, returns a short trimmed version of the input as a fallback.
    """
    prompt = f"Summarize this research abstract for a weekly digest (3–4 sentences, technical tone):\n\n{text}"
    if not OPENROUTER_KEY:
        # Fallback: return first 2-3 sentences from the abstract
        return " ".join(re.split(r"(?<=[.!?])\\s+", text)[:3]).strip()

    headers = {"Authorization": f"Bearer {OPENROUTER_KEY}", "Content-Type": "application/json"}
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.5,
        "max_tokens": 300,
    }
    try:
        resp = requests.post(OPENROUTER_URL, json=body, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        # handle OpenAI-like response shape
        choice = data.get("choices", [])[0]
        # some providers return message.content, others return text
        content = None
        if isinstance(choice, dict):
            msg = choice.get("message") or {}
            content = msg.get("content") or choice.get("text")
        if not content:
            content = str(choice)
        return content.strip()
    except Exception:
        return " ".join(re.split(r"(?<=[.!?])\\s+", text)[:3]).strip()

def generate_digest(papers):
    """Generate Markdown digest text."""
    today = date.today().isoformat()
    lines = [f"# Weekly Digest: {TOPIC} ({today})\n"]
    for i, paper in enumerate(papers, 1):
        summary = summarize_text(paper["summary"])
        lines.append(f"### {i}. {paper['title']}")
        lines.append(f"**Authors:** {paper['authors']}")
        lines.append(f"**Summary:** {summary}")
        lines.append(f"[Read on arXiv]({paper['link']})\n")
    return "\n".join(lines)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    papers = search_arxiv(TOPIC, MAX_RESULTS)
    digest_md = generate_digest(papers)
    filename = f"{OUTPUT_DIR}/digest_{date.today()}.md"
    with open(filename, "w") as f:
        f.write(digest_md)
    print(f"Digest saved to {filename}")

if __name__ == "__main__":
    main()
