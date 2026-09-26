"""
news_service.py

Fetches top headlines from NewsAPI.org (https://newsapi.org) -- that part
is plain Python (get_headlines), and always will be: the model never picks
which headlines exist, so it can't drop a real one or invent one that
wasn't reported.

Turning those headlines into a spoken brief (build_news_brief) is then
handed to the small local model as a narrow rewrite job: it's given the
exact, already-fetched headlines and told to only narrate them naturally,
never add/drop/invent one. _looks_complete double-checks its output still
mentions every headline before trusting it; if the model call fails, or
the check fails, build_news_brief falls back to the old fixed
opener/transition template (_template_brief), so a bad model response is
never what gets said out loud.

Setup: get a free key at https://newsapi.org/register, then either set a
real NEWS_API_KEY environment variable, or put a line

    NEWS_API_KEY=your-key-here

in a `.env` file in this folder. `.env` is already gitignored, so the key
never gets committed. No extra dependency for that -- `_load_dotenv` below
is a deliberately tiny reader, not python-dotenv, to keep the app installable
on a Pi with nothing beyond requirements.txt.
"""

import os
import random
import re
import time

import requests

from llm_client import chat_completion

TOP_HEADLINES_URL = "https://newsapi.org/v2/top-headlines"
DEFAULT_COUNTRY = "us"
DEFAULT_PAGE_SIZE = 5
REQUEST_TIMEOUT = 6

# Headlines don't change minute to minute, and NewsAPI's free tier caps you
# at 100 requests/day, so repeated "what's the news" within this window
# reuses the last fetch instead of spending another call.
CACHE_SECONDS = 10 * 60

_cache = {}  # (country, category, page_size) -> (fetched_at, headlines)


def _load_dotenv():
    """A minimal .env reader: KEY=VALUE per line, '#' comments and blank
    lines ignored, surrounding quotes stripped. Never overwrites a real
    environment variable that's already set."""
    path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


_load_dotenv()

API_KEY = os.environ.get("NEWS_API_KEY")


class NewsUnavailable(Exception):
    """Headlines couldn't be fetched -- no key, network down, or the API
    itself failed. The message is short and spoken to the user as-is;
    the technical detail goes to the console instead (see below)."""


def _clean_title(title, source):
    """NewsAPI headlines usually end " - <source name>", which is redundant
    once we say the source separately ("...from BBC News. ...from BBC
    News."). Only strips it when it's actually that exact suffix, so a
    headline that legitimately contains " - " keeps it."""
    suffix = f" - {source}"
    return title[: -len(suffix)] if title.endswith(suffix) else title


def get_headlines(country=DEFAULT_COUNTRY, category=None, page_size=DEFAULT_PAGE_SIZE):
    """Top headlines right now, ranked as NewsAPI returns them. Returns a
    list of {"title", "source"} dicts, up to `page_size` of them (fewer if
    that's all there is). Cached for CACHE_SECONDS. Raises NewsUnavailable
    if there's no key, the request fails, or the API reports an error."""
    if not API_KEY:
        print("  [news] no NEWS_API_KEY set -- get a free key at "
              "https://newsapi.org/register and add NEWS_API_KEY=... to a .env file")
        raise NewsUnavailable("I don't have a news API key set up yet, sir.")

    cache_key = (country, category, page_size)
    cached = _cache.get(cache_key)
    if cached and time.monotonic() - cached[0] < CACHE_SECONDS:
        return cached[1]

    params = {"apiKey": API_KEY, "country": country, "pageSize": page_size}
    if category:
        params["category"] = category

    try:
        response = requests.get(TOP_HEADLINES_URL, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        print(f"  [news] request failed: {exc}")
        raise NewsUnavailable("I couldn't reach the news service, sir.") from exc

    if payload.get("status") != "ok":
        print(f"  [news] API error: {payload.get('message')}")
        raise NewsUnavailable("The news service ran into a problem, sir.")

    headlines = [
        {"title": _clean_title(a["title"], a["source"]["name"]), "source": a["source"]["name"]}
        for a in payload.get("articles", [])
        if a.get("title") and a["title"] != "[Removed]"
    ]
    _cache[cache_key] = (time.monotonic(), headlines)
    return headlines


OPENERS = [
    "Here's the news, sir.",
    "Here's what's happening, sir.",
    "Sir, here are today's top stories.",
    "Here's your news brief, sir.",
]

TRANSITIONS = ["Also,", "Meanwhile,", "In other news,", "Elsewhere,", "Next,"]

NO_HEADLINES_LINE = "There don't seem to be any headlines right now, sir."


def _template_brief(headlines):
    """The old, fully deterministic delivery -- opener, then each headline
    with a random transition and its source. Used whenever the model isn't
    available to narrate, or its narration doesn't check out."""
    parts = [random.choice(OPENERS)]
    for i, headline in enumerate(headlines):
        line = f"{headline['title']}, from {headline['source']}"
        if i > 0:
            line = f"{random.choice(TRANSITIONS)} {line}"
        parts.append(line + ".")
    return " ".join(parts)


def _looks_complete(summary, headlines):
    """True if `summary` still mentions -- in some form -- every headline
    it was given. Catches the model dropping, merging, or garbling a story,
    which matters more here than sounding natural: a headline that quietly
    goes missing is worse than a slightly stiff delivery."""
    lowered = summary.lower()

    def _mentioned(headline):
        words = [w for w in re.findall(r"[a-z']+", headline["title"].lower()) if len(w) > 3]
        if not words:
            return True
        hits = sum(1 for w in words if w in lowered)
        return hits >= max(1, len(words) // 2)

    return all(_mentioned(h) for h in headlines)


# With exactly one headline the model reliably invented a second, fake
# "story" restating the same one, even when explicitly told not to -- a
# 1.7B model doesn't hold that instruction well. There's also nothing a
# narration adds when there's only one line to say, so that case is never
# sent to the model at all; _template_brief handles it (and always will).
_MIN_HEADLINES_TO_SUMMARIZE = 2


def _summarize_headlines(headlines):
    """Ask the small local model to read these EXACT headlines out loud
    naturally, instead of the fixed template. It's given the real
    headlines and sources verbatim and told to only narrate them, not
    add commentary or invent anything -- the same kind of narrow rewrite
    job as extract_event_title, just for delivery instead of extraction.
    Returns None (falls back to _template_brief) if there's nothing worth
    summarizing, the call fails, or the result doesn't check out against
    _looks_complete."""
    if len(headlines) < _MIN_HEADLINES_TO_SUMMARIZE:
        return None

    numbered = "\n".join(
        f"{i + 1}. {h['title']} (source: {h['source']})" for i, h in enumerate(headlines)
    )
    try:
        message = chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a voice assistant reading today's top news "
                        "headlines out loud. Below are the REAL headlines, "
                        "already fetched -- this is ALL the information you "
                        "have; you were NOT given any article body or extra "
                        "facts. Use ONLY the words in these headlines. Do not "
                        "add, invent, or guess ANY fact, number, cause, "
                        "consequence, or detail that is not literally present "
                        "in the headline text, not even a plausible-sounding "
                        "one. Your only job is to smooth the DELIVERY: "
                        "rephrase each headline into one natural spoken "
                        "sentence, add a natural spoken transition before "
                        "every headline after the first one (\"Meanwhile,\" "
                        "\"Also,\" \"In other news,\" ...), and always name "
                        "that story's source. Do not merge or drop a story, "
                        "and do not add commentary/opinion. If unsure whether "
                        "something counts as adding information, leave it "
                        "out. Output ONLY the narration -- no headers, no "
                        "numbered list, no markdown, no quotes.\n\n"
                        "Example input:\n"
                        "1. City council approves new downtown park (source: Local News)\n"
                        "2. Tech company unveils new smartphone (source: TechDaily)\n\n"
                        "Example output:\n"
                        "The city council has approved a new downtown park, "
                        "according to Local News. Meanwhile, a tech company "
                        "has unveiled a new smartphone, reports TechDaily."
                    ),
                },
                {"role": "user", "content": numbered},
            ],
            temperature=0.2,
        )
    except Exception as exc:
        print(f"  [news] summarizer call failed: {exc}")
        return None

    summary = message["content"].strip()
    if summary and _looks_complete(summary, headlines):
        return summary
    print(f"  [news] summarizer output didn't check out, falling back: {summary!r}")
    return None


def build_news_brief(country=DEFAULT_COUNTRY, category=None, page_size=DEFAULT_PAGE_SIZE):
    """The spoken news brief, or a short apology if headlines can't be
    fetched -- never raises, so callers don't need their own try/except
    for the everyday failure modes (no key yet, network hiccup, API down)."""
    try:
        headlines = get_headlines(country=country, category=category, page_size=page_size)
    except NewsUnavailable as exc:
        return str(exc)

    if not headlines:
        return NO_HEADLINES_LINE

    return _summarize_headlines(headlines) or _template_brief(headlines)


if __name__ == "__main__":
    print(build_news_brief())
