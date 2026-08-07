# news_pipeline.py
# Fetches copper-related news, analyzes sentiment and relevance via local Ollama LLM,
# and stores results in PostgreSQL.

import json
import feedparser
import requests
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime, timezone
from urllib.request import Request, urlopen
import os
from dotenv import load_dotenv
load_dotenv() 
 
# --- Configuration ---
DB_CONFIG = {
    "dbname": os.getenv("POSTGRES_DB", "copper_db"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", ""),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
}

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

RSS_FEEDS = [
    "https://www.mining-technology.com/feed/",
    "https://www.proactiveinvestors.com/rss/news.rss",
    "https://www.resourceworld.com/feed/",
    "https://news.google.com/rss/search?q=copper+price+LME&hl=en&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=copper+mining+production&hl=en&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=copper+demand+China&hl=en&gl=US&ceid=US:en",
]



CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS news_articles (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    summary TEXT,
    url TEXT UNIQUE,
    source TEXT,
    published_at TIMESTAMP,
    sentiment VARCHAR(10),       -- positive / negative / neutral
    sentiment_score FLOAT,       -- -1.0 to 1.0
    relevance_score FLOAT,       -- 0.0 to 1.0

    llm_summary TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);
"""

# --- Fetch ---
def fetch_feed(url: str) -> list[dict]:
    """Download and parse a single RSS feed, return list of article dicts."""
    try:
        req = Request(url, headers=HEADERS)
        with urlopen(req, timeout=20) as resp:
            feed = feedparser.parse(resp.read())
        source = feed.feed.get("title") or url or "unknown"
        articles = []
        for entry in feed.entries:
            articles.append({
                "title": entry.get("title", "").strip(),
                "summary": entry.get("summary", "").strip(),
                "url": entry.get("link", ""),
                "source": source,
                "published_at": entry.get("published", datetime.now(timezone.utc).isoformat()),
            })
        return articles
    except Exception as e:
        print(f"[WARN] Feed error ({url}): {e}")
        return []

def fetch_newsapi_articles(query="copper price OR copper mining"):
    resp = requests.get(
        "https://newsapi.org/v2/everything",
        params={"q": query, "sortBy": "publishedAt", "pageSize": 50, "apiKey": os.getenv("NEWSAPI_KEY", "")},
        timeout=10
    )
    articles = resp.json().get("articles", [])
    return [
        {
            "title": a["title"],
            "summary": a.get("description", ""),
            "url": a["url"],
            "source": a["source"]["name"],
            "published_at": a["publishedAt"],
        }
        for a in articles if a.get("title")
    ]


def fetch_all_news() -> list[dict]:
    articles = []
    seen_urls = set()
    for url in RSS_FEEDS:
        for article in fetch_feed(url):
            if article["url"] not in seen_urls:
                seen_urls.add(article["url"])
                articles.append(article)
    for article in fetch_newsapi_articles():
        if article["url"] not in seen_urls:
            seen_urls.add(article["url"])
            articles.append(article)
    print(f"[INFO] Fetched {len(articles)} unique articles.")
    return articles


# --- LLM Analysis ---
#PROMPT_TEMPLATE = """Analyze this news article about commodities/copper markets.
#Return ONLY valid JSON with these fields:
#- "summary": one sentence summary (max 100 words)
#- "sentiment": one of "positive", "negative", "neutral"
#- "sentiment_score": float from -1.0 (very negative) to 1.0 (very positive)
#- "relevance_score": float from 0.0 to 1.0 (how relevant to copper market)

#Title: {title}
#Content: {summary}

#JSON:"""
PROMPT_TEMPLATE = """Analyze this news article about copper/commodities markets.
Return ONLY valid JSON with these fields:
- "summary": one sentence (max 100 words)
- "sentiment": "positive", "negative", or "neutral" — reflects the PRICE DIRECTION bias
    (positive = bullish for copper price, negative = bearish, neutral = no clear directional bias)
    NOT the emotional tone of the article itself.
- "sentiment_score": float from -1.0 to 1.0 representing the EXPECTED IMPACT ON COPPER PRICE:
    +1.0 = strongly bullish (supply disruption, demand surge, inventory drop, weak USD)
    -1.0 = strongly bearish (demand slowdown, oversupply, recession fears, strong USD)
     0.0 = no clear price impact
- "relevance_score": float from 0.0 to 1.0 (how directly this affects copper supply/demand/price)

Title: {title}
Content: {summary}

JSON:"""


def analyze_with_ollama(article: dict) -> dict:
    """Send article to local Ollama and parse JSON response."""

    prompt = PROMPT_TEMPLATE.format(
        title=article["title"],
        summary=article["summary"][:500]  # limit tokens
    )
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=60
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "")
        # Extract JSON block if wrapped in markdown
        if "```" in raw:
           raw = raw.split("```")[1].strip().lstrip("json").strip()
        result = json.loads(raw)
        return {
            "llm_summary": result.get("summary", ""),
            "sentiment": result.get("sentiment", "neutral"),
            "sentiment_score": float(result.get("sentiment_score", 0.0)),
            "relevance_score": float(result.get("relevance_score", 0.0)),
        }
    except Exception as e:
        print(f"[WARN] Ollama error for '{article['title'][:50]}': {e}")
        return {"llm_summary": "", "sentiment": "neutral",
                "sentiment_score": 0.0, "relevance_score": 0.0}


# --- Database ---
def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def init_db(conn):
    with conn.cursor() as cur:
        cur.execute(CREATE_TABLE_SQL)
    conn.commit()


def load_articles(conn, articles: list[dict]):
    """Insert articles, skip duplicates by URL."""
    if not articles:
        return
    rows = [(
        a["title"], a["summary"], a["url"], a["source"],
        a.get("published_at"), a.get("sentiment"), a.get("sentiment_score"),
        a.get("relevance_score"), a.get("llm_summary")
    ) for a in articles]

    sql = """
        INSERT INTO news_articles
            (title, summary, url, source, published_at,
             sentiment, sentiment_score, relevance_score, llm_summary)
        VALUES %s
        ON CONFLICT (url) DO NOTHING;
    """
    with conn.cursor() as cur:
        execute_values(cur, sql, rows)
    conn.commit()
    print(f"[INFO] Inserted/skipped {len(rows)} articles.")


# --- Pipeline ---
def run():
    articles = fetch_all_news()

    # Filter low-relevance before hitting LLM (quick keyword pre-filter)
    copper_keywords = {"copper", "lme", "comex", "cathode", "concentrate", "smelter"}
    candidates = [
        a for a in articles
        if any(kw in (a["title"] + a["summary"]).lower() for kw in copper_keywords)
    ]
    print(f"[INFO] {len(candidates)} articles passed keyword filter, analyzing...")

    for article in candidates:
        analysis = analyze_with_ollama(article)
        article.update(analysis)

    conn = get_connection()
    try:
        init_db(conn)
        load_articles(conn, candidates)
    finally:
        conn.close()
    print("[INFO] Pipeline complete.")


if __name__ == "__main__":
    run()

