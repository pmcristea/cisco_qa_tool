import logging

import langchain

import streamlit as st

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(levelname)s:%(name)s:%(funcName)s:%(message)s")

file_handler = logging.FileHandler('qa_tools.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

def return_new_links(links: list, 
                     vectordb: langchain.vectorstores.pinecone.Pinecone) -> list:
    """Checks an existing vectordb whether it contains any documents having URLs matching those in links.  Any URLs found to match are removed as they imply the document already exists in the vectordb and doesn't need to be added again.

    Args:
        links: a list of URLs
        vectordb: a Pinecone vectordatabase 
    
    Returns:
        new_links: a list of URLs not already present in vectordb.
    """
    
    new_links = []
    
    num_initial_links = len(links)
    logger.info(f"{num_initial_links} links.  Checking which links are already connected to a PDF in the vectordb.")
    for url in links:
        if url.endswith('.pdf'):
            pdf_html_path = url
        else:
            pdf_html_path = url.split(".html", 1)[0] + ".pdf"
        
        # Find which urls in links match with PDF files already in the vectordb
        matches = vectordb.similarity_search(query=' ', 
                                             k=1, 
                                             filter={'source':pdf_html_path})
        
        # If any matches are found, it implies the PDF file exists in the vectordb and we can ignore the url.  Continue the loop and move onto the next url
        if len(matches)>0:
            continue
        
         # If no matches are found, keep the URL so the PDF file can be added to vectordb later
        else:
            logger.debug(f'{url} not found in database')
            new_links.append(url)
    
    num_new_links = len(new_links)
    logger.info(f"{num_new_links} links not connected to any PDFs in the vectordb.")
    
    return new_links