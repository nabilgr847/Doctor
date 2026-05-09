import os
import requests
from ddgs import DDGS
from serpapi import GoogleSearch
from gdeltdoc import GdeltDoc, Filters

def search_all_unrestricted():
    results = []
    # 1. DuckDuckGo
    try:
        ddgs = DDGS()
        results.append(str(ddgs.text("cancer immunotherapy latest research", max_results=5)))
    except:
        pass
    # 2. GDELT
    try:
        gd = GdeltDoc()
        f = Filters(keyword="cancer", start_date="2026-05-01", end_date="2026-05-09")
        articles = gd.article_search(f)
        results.append(str(articles.head(5).to_dict()))
    except:
        pass
    # 3. OpenAlex
    try:
        oa = requests.get("https://api.openalex.org/works", params={"search": "cancer", "per_page": 5}).json()
        results.append(str(oa))
    except:
        pass
    # 4. Semantic Scholar
    try:
        ss = requests.get("https://api.semanticscholar.org/graph/v1/paper/search", params={"query": "cancer", "limit": 5}).json()
        results.append(str(ss))
    except:
        pass
    return "\n\n".join(results)

def search_serpapi():
    try:
        key = os.getenv("SERPAPI_KEY")
        if not key:
            return ""
        serp = GoogleSearch({"q": "cancer immunotherapy breakthroughs", "api_key": key})
        out = serp.get_dict()
        return str(out.get("organic_results", []))
    except:
        return ""
