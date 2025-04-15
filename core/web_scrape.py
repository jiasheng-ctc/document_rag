import requests
import time
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# User agent to mimic a browser
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

def search_duckduckgo(query: str, num_results: int = 2) -> List[str]:
    """
    Perform a search using DuckDuckGo and extract URLs from the results.
    
    Args:
        query (str): The search query
        num_results (int): Number of results to return
        
    Returns:
        List[str]: List of URLs from search results
    """
    # DuckDuckGo search URL
    search_url = f"https://html.duckduckgo.com/html/?q={query}"
    
    try:
        logger.info(f"Searching for: {query}")
        response = requests.get(search_url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        results = soup.find_all('a', class_='result__url')
        
        urls = []
        for result in results[:num_results]:
            href = result.get('href')
            if href and href.startswith('http'):
                urls.append(href)
            elif href:
                # Some URLs might be relative
                urls.append(f"https:{href}" if href.startswith('//') else f"https://{href}")
        
        logger.info(f"Found {len(urls)} URLs")
        return urls
        
    except Exception as e:
        logger.error(f"Error searching DuckDuckGo: {e}")
        return []

def scrape_webpage(url: str, timeout: int = 10, max_length: int = 1000) -> str:
    """
    Scrape content from a webpage with content length limit.
    
    Args:
        url (str): The URL to scrape
        timeout (int): Request timeout in seconds
        max_length (int): Maximum length of content to return
        
    Returns:
        str: The extracted text content
    """
    try:
        logger.info(f"Scraping URL: {url}")
        response = requests.get(url, headers=HEADERS, timeout=timeout)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
            
        # Extract text content
        text = soup.get_text(separator='\n')
        
        # Clean up text: remove extra newlines and spaces
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        # Return a reasonably sized excerpt
        return text[:max_length]
        
    except Exception as e:
        logger.error(f"Error scraping {url}: {e}")
        return f"[Error scraping {url}: {str(e)}]"

def scrape_web(query: str, max_results: int = 2, max_content_per_site: int = 1000) -> str:
    """
    Perform a web search and scrape the content from the top search results.

    Args:
        query (str): The search query to use for web scraping.
        max_results (int): Maximum number of results to scrape
        max_content_per_site (int): Maximum content length per site

    Returns:
        str: The combined text content of the scraped search results.
    """
    logger.info(f"Starting web search for: '{query}'")
    
    # Limit query length
    if len(query) > 100:
        query = query[:100]
    
    # Search for relevant URLs
    urls = search_duckduckgo(query, num_results=max_results)
    
    if not urls:
        return "No relevant information found on the web for this query."
    
    results = []
    
    # Scrape each URL and append the result to the output
    for url in urls:
        result = scrape_webpage(url, max_length=max_content_per_site)
        if result:
            results.append(f"Source: {url}\n\n{result}")
        
        # Be polite with rate limiting
        time.sleep(1)
    
    logger.info(f"Completed scraping {len(results)} URLs")
    
    # Combine results with source attribution
    if results:
        combined = "\n\n---\n\n".join(results)
        # Limit total combined length
        if len(combined) > 2000:
            combined = combined[:2000] + "... [content truncated]"
        return combined
    else:
        return "Unable to retrieve relevant information from the web for this query."