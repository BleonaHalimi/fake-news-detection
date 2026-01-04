"""
Web Scraper Utilities
Extracts article content from URLs
"""

import requests
from bs4 import BeautifulSoup
import validators


def is_valid_url(url):
    """
    Check if a URL is valid

    Args:
        url (str): URL to validate

    Returns:
        bool: True if valid, False otherwise
    """
    return validators.url(url) is True


def extract_article_from_url(url):
    """
    Extract article content from a URL
    Uses BeautifulSoup for parsing

    Args:
        url (str): URL of the article

    Returns:
        tuple: (article_data: dict or None, error: str or None)
            article_data = {
                'title': article title,
                'text': article content,
                'url': original URL
            }
    """
    try:
        # Validate URL
        if not is_valid_url(url):
            return None, "Invalid URL format. Please enter a valid URL starting with http:// or https://"

        # Set comprehensive headers to mimic a real browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0'
        }

        # Fetch the page
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract title
        title = ""
        title_tag = soup.find('title')
        if title_tag:
            title = title_tag.get_text().strip()

        # Also try h1 tags
        if not title:
            h1_tag = soup.find('h1')
            if h1_tag:
                title = h1_tag.get_text().strip()

        # Extract article text
        # Try common article containers
        article_text = ""

        # Strategy 1: Look for <article> tag
        article_tag = soup.find('article')
        if article_tag:
            article_text = article_tag.get_text(separator=' ', strip=True)

        # Strategy 2: Look for common article class names
        if not article_text:
            for class_name in ['article-body', 'article-content', 'story-body',
                              'post-content', 'entry-content', 'content']:
                content_div = soup.find('div', class_=class_name)
                if content_div:
                    article_text = content_div.get_text(separator=' ', strip=True)
                    break

        # Strategy 3: Look for paragraphs in main content area
        if not article_text:
            main_tag = soup.find('main') or soup.find('div', {'id': 'content'})
            if main_tag:
                paragraphs = main_tag.find_all('p')
                article_text = ' '.join([p.get_text(strip=True) for p in paragraphs])

        # Strategy 4: Fallback - get all paragraphs
        if not article_text:
            paragraphs = soup.find_all('p')
            article_text = ' '.join([p.get_text(strip=True) for p in paragraphs])

        # Clean up text
        article_text = ' '.join(article_text.split())  # Remove extra whitespace

        if not article_text or len(article_text) < 100:
            return None, "Could not extract sufficient article content from this URL. Please try copying the text manually."

        return {
            'title': title,
            'text': article_text,
            'url': url
        }, None

    except requests.exceptions.Timeout:
        return None, "Request timed out. Please try again."

    except requests.exceptions.ConnectionError:
        return None, "Could not connect to the URL. Please check your internet connection."

    except requests.exceptions.HTTPError as e:
        if '403' in str(e) or '401' in str(e):
            return None, (
                "This website blocks automated access. Please:\n"
                "1. Open the article in your browser\n"
                "2. Copy the article text\n"
                "3. Use the 'Single Analysis' page to paste and analyze it\n\n"
                f"Technical error: {e}"
            )
        return None, f"HTTP error occurred: {e}"

    except Exception as e:
        return None, f"An error occurred while extracting the article: {str(e)}"


def extract_with_newspaper3k(url):
    """
    Alternative extraction method using newspaper3k library
    Note: newspaper3k might need to be imported separately

    Args:
        url (str): URL of the article

    Returns:
        tuple: (article_data: dict or None, error: str or None)
    """
    try:
        from newspaper import Article

        article = Article(url)
        article.download()
        article.parse()

        return {
            'title': article.title,
            'text': article.text,
            'url': url,
            'authors': article.authors,
            'publish_date': str(article.publish_date) if article.publish_date else None
        }, None

    except ImportError:
        return None, "newspaper3k library not installed. Using fallback method."

    except Exception as e:
        return None, f"Could not extract article: {str(e)}"
