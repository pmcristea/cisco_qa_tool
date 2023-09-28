import logging

from requests_html import HTMLResponse

import streamlit as st

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(levelname)s:%(name)s:%(funcName)s:%(message)s")

file_handler = logging.FileHandler('web_tools.log')
# file_handler.setLevel(logging.ERROR)
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

def get_source(url: str, PROXIES: dict = None) -> HTMLResponse:
    """Return the HTML source for the provided URL.
    
    Args:
        url: URL to extract HTML source from.
        PROXIES: dictionary of proxy IP addresses to use for HTTP requests
        
    Returns:
        response: HTMLResponse object containing HTML source.
    """  
    
    from fake_useragent import UserAgent
    
    import requests
    from requests.adapters import HTTPAdapter
    
    from requests_html import HTMLSession
    
    from urllib3.util.retry import Retry
    
    # Randomly generate User-Agent headers to mask HTTP get requests to look like a user and not a bot
    headers = {}
    headers["User-Agent"] = UserAgent().random
    
    # Initiate HTML session and set rules for retrying HTTP get requests upon failure
    session = HTMLSession()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    # Initiate HTTP get request
    response = session.get(url, headers=headers, proxies=PROXIES)
    
    # Log results
    if response.ok:
        logger.info(f"Response for {url} successfully received.")
    else:
        logger.info(f"Response for {url} failed.")
    
    return response



def scrape_google(website_url: str, 
                  product_name: str, 
                  additional_search_text: str = 'none',
                  language_code: str = 'en',
                  PROXIES: dict = None) -> list:
    """Return a list of URLs corresponding to Google search results from a particular website.  Results from certain URLs that won't contain relevant information are excluded based on the URL's prefixes and suffixes.

    Args:
        website_url: URL for Google to focus its search on.  Example: www.exampleurl.com
        product_name: Name of the product to search for information on.
        additional_search_text: Any additional text to include in the search that could help refine the results.
        language_code: Specifies which language to return search results in.
        PROXIES: proxy IP addresses to use for HTTP requests

    Returns:
        links: A list of URLs corresponding to search results from Google.
    """
    import urllib
    
    # Combine user input to construct the query used for the Google search
    search_args = [website_url, product_name, additional_search_text]
    joined_search_args = " ".join(search_args)
    query_suffix = urllib.parse.quote_plus(joined_search_args)  
    query = f"https://www.google.com/search?q=site%3A{query_suffix}&lr=lang_{language_code}"
    
    logger.debug(f'query_suffix:{query_suffix}')
    logger.debug(f'query:{query}')
    
    # Initiate HTTP get request and scrape the URLs of the links corresponding to the search results
    response = get_source(query, PROXIES)
    links = list(response.html.absolute_links)
    
    # Establish criteria to remove search results corresponding to Google domains that contain information we don't care about
    google_domains = ('https://www.google.', 
                      'https://google.', 
                      'https://webcache.googleusercontent.', 
                      'http://webcache.googleusercontent.', 
                      'https://policies.google.',
                      'https://support.google.',
                      'https://maps.google.',
                      'https://translate.google.')
    
    # Placeholder for if suffixes ever become useful to detect results we don't want 
    suffixes = ()
    
    init_num_links = len(links)
    
    st.write(f"Initial search found {init_num_links} links that could contain the answer.  Now identifying and removing ones that aren't useful.")
    logger.info(f"{init_num_links} intial links found.")
    logger.debug(f"Initial links are: {links}")
    
    # Remove search results based on the criteria established earlier
    for url in links[:]:
        if url.startswith(google_domains):
            links.remove(url)
        if url.endswith(suffixes):
            links.remove(url)
    
    # Sort remaining links
    links.sort()
    
    new_num_links=len(links)
    
    st.write(f"{new_num_links} useful links remain.")
    logger.info(f"{new_num_links} useful links remain.")
    logger.debug(f"Remaining useful links:{links}")
    
    return links