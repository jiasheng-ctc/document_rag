import requests
from googlesearch import search


def scrape_jina_ai(url: str) -> str:
    """
    Scrape the Jina AI redirection service for a given URL.

    Args:
        url (str): The URL to scrape.

    Returns:
        str: The text content of the scraped page.
    """
    response = requests.get("https://r.jina.ai/" + url)
    return response.text


def scrape_web(query: str) -> str:
    """
    Perform a web search and scrape the content from the top search results.

    Args:
        query (str): The search query to use for web scraping.

    Returns:
        str: The combined text content of the scraped search results.
    """
    print(f"🔍 Performing web search for query: '{query}'")
    urls = search(query, num_results=1)
    results = ""

    # Scrape each URL and append the result to the output
    for url in urls:
        print(f"🌐 Scraping URL: {url}")
        result = scrape_jina_ai(url)
        results += result + "\n"

    print("✅ Finished scraping URLs.")
    return results
