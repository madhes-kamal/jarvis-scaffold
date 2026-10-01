"""
web_search.py

General web search, exposed to the model as a tool (see conversation.py)
for anything it doesn't already know or that needs to be current --
"what's the weather in Tokyo", "who won the game last night", "what's
the latest on X".

Backed by DuckDuckGo via the `ddgs` package: no API key, no sign-up,
unlike news_service.py's NewsAPI. The tradeoff is it's an unofficial
scrape of DuckDuckGo's search results rather than a stable published
API, so it can break or get rate-limited without warning -- if that
becomes a problem, swap this file for a paid provider (e.g. Tavily)
without touching conversation.py, same as llm_client.py lets the model
backend change underneath everything else.

The model never sees raw HTML or ranking internals -- just a short list
of title/snippet/url per result, plain data it reads and summarizes in
its own words, the same shape as get_upcoming_events.
"""

from ddgs import DDGS

DEFAULT_MAX_RESULTS = 5
REQUEST_TIMEOUT = 8


class WebSearchUnavailable(Exception):
    """The search couldn't be completed -- network down, DuckDuckGo
    rate-limited us, or it returned nothing usable. The message is short
    and safe to speak as-is."""


def web_search(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> list:
    """Search the web and return the top results for a query. Use this for
    anything current, factual, or outside your own knowledge -- weather,
    news events, scores, prices, "who is/what is" questions about things
    that might have changed since you were trained.

    Args:
        query: What to search for, as a short plain-text search query.
        max_results: How many results to return (default 5).
    """
    try:
        results = DDGS().text(query, max_results=max_results, timeout=REQUEST_TIMEOUT)
    except Exception as exc:
        print(f"  [web_search] request failed: {exc}")
        raise WebSearchUnavailable("I couldn't search the web just now, sir.") from exc

    if not results:
        raise WebSearchUnavailable("I didn't find anything for that, sir.")

    return [
        {"title": r.get("title", ""), "snippet": r.get("body", ""), "url": r.get("href", "")}
        for r in results
    ]


if __name__ == "__main__":
    for item in web_search("current weather in Tokyo"):
        print(item)
