import logging

import langchain

import streamlit as st

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(levelname)s:%(name)s:%(funcName)s:%(message)s")

file_handler = logging.FileHandler('pdf_tools.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)



def load_pdf_as_doc(html_path: str, 
                    local_path: str, 
                    filename: str) -> list:
    """Loads PDF file from local_path and generates metadata for each page of the document.

    Args:
        html_path: URL where the PDF file was originally downloaded from
        local_path: File path used to read the PDF file from local disk
        filename: filename of PDF 

    Returns:
        pdf_doc: A list where each element is one page of the PDF file as a Langchain Document object
    """
    from langchain.document_loaders import PyPDFLoader
    
    # Load local PDF file into memory 
    loader = PyPDFLoader(file_path=local_path)
    
    # Create a Langchain Document object for each page in the PDF file
    pdf_doc = loader.load()
    
    num_pages = len(pdf_doc)
    
    logger.info(f'Loading {num_pages} page document')
    
    # Add the same metadata to each page of the PDF document
    for page in pdf_doc:
        page.metadata['source'] = html_path
        page.metadata['filename'] = filename
        
        # Langchain uses 0 indexing to reference the first page of a document.  This increases the page index by 1 so the first page is 1, 2nd page is 2, etc.
        page.metadata['page'] = int(page.metadata['page']) + 1
    
    logger.info(f'All pages loaded.')
    
    return pdf_doc



def download_pdf_and_return_doc(html_path: str, 
                                pdf_filename: str, 
                                proxies=None) -> list[langchain.schema.document.Document]:
    """
    Initiates HTTP get request for the URL of a pdf file.  Writes the contents of the PDF file to local disk.

    Args:
        html_path: HTML URL of a PDF file
        pdf_filename: Filename of PDF file
        proxies: proxy IP addresses to use for HTTP requests

    Returns:
        pdf_doc: A list where each element is one page of the PDF file as a Langchain Document object
    """
    
    import tempfile
    import pathlib
    
    from web_tools import get_source
    
    # Create local directory to temporarily store downloaded PDF file
    temp_dir = tempfile.TemporaryDirectory()
    temp_dir_path = pathlib.Path(temp_dir.name)
    temp_file_path = temp_dir_path / pdf_filename
    temp_file_path_string = str(temp_file_path)
    
    logger.debug(f"Temp directory created at: {str(temp_dir_path)}")
    
    # Get HTTP response of URL
    response = get_source(html_path, proxies)
    
    # Check to see if the HTTP response is actually for a PDF file
    response_content_type = response.headers['Content-Type']
    logger.debug(f'Response content type: {response_content_type}')
    
    if response_content_type == 'application/pdf':
        # If the file is a PDF, write the contents to local_path            
        with open(temp_file_path, 'wb') as pdf:
            pdf.write(response.content)
            pdf_doc = load_pdf_as_doc(html_path, temp_file_path_string, pdf_filename)
            
        logger.debug(f"{temp_file_path} successfully written to disk.")
            
        return pdf_doc
    
    else:
        logger.debug(f"{html_path} didn't contain a PDF file")
        
        

def generate_cisco_metadata(pdf_doc: list) -> list[langchain.schema.document.Document]:
    """Extracts information from the URL of a data sheet pulled from www.cisco.com to be used as metadata in the document.

    Args:
        pdf_doc: A list where each element is a single page of a pdf file

    Returns:
        new_pages: new pages of pdf_doc after attempting to add metadata.  If no metadata could be added, the original pages of the pdf_doc are returned.
    """
    
    new_pages = []
    
    for page in pdf_doc:  
    # Many source URLs for Cisco datasheets follow this structure:
    # https://www.cisco.com/.../collateral/product_category/product_name/filename.pdf
    # The product_category, product_name, and filename are useful and can be extracted
    # and used as metadata

    # Separates the URL into a list of the form:
    # ['https:', '...', 'www.cisco.com', '...', 'collateral', 'product_category', 'product_name', 'filename']
        pdf_html_path = page.metadata['source']
        split_url = pdf_html_path.split('/')

        try:
            # Find the index of "collateral" in the split_url and set metadata based on that index
            idx = split_url.index('collateral')

            product_category = split_url[idx+1]
            product_name = split_url[idx+2]

            page.metadata["product_category"] = product_category
            page.metadata["product_name"] = product_name

        except ValueError as e:
            # If "collateral" isn't present in the split_url, break the loop.  All pages in the document come from the same url and there's no need to try and generate metadata for other pages.
            logger.exception(f"Failed to get metadata.  Keyword 'collateral' couldn't be found in {pdf_html_path}")           
            break
        
        else:
            new_pages.append(page)
    
    # If any new pages were generated because metadata was found, return them.  Otherwise, return the original pdf_doc
    if len(new_pages)>0:
        logger.debug(f"""Product category: {pdf_doc[0].metadata["product_category"]}, \nProduct_name: {pdf_doc[0].metadata["product_name"]}, \nMetadata added for all pages in {pdf_doc[0].metadata["filename"]}.""")
        return new_pages
    
    else:
        return pdf_doc
    
    
    
def return_pdf_docs(links: list, 
                    proxies: dict = None,
                    is_cisco_datasheet: bool = False) -> list[langchain.schema.document.Document]:
    """
    Download PDFs from URLs, load from local disk as Langchain Documents and add metadata.  Filters URLs so that duplicate PDF files are ignored.

    Args:
        links: A list of URLs pointing towards possible PDF files
        proxies: proxy IP addresses to use for HTTP requests
        is_cisco_datasheet: If True, extracts information from the URL to be used as metadata based on criteria specific to URLs containing for Cisco data sheets

    Returns:
        unique_pdf_docs: A list of Langchain PDF documents
    """
    

    unique_pdf_docs = []
    unique_pdf_paths = []

    for link in links:
        # Cisco datasheets are available as both a HTML file and a PDF file.  Any datasheet that has a URL ending with .html can be accessed from the same URL if the .html is replaced with .pdf.  
        if link.endswith(".pdf"):
            pdf_html_path = link
            logger.debug(f"URL already links to PDF document: {pdf_html_path}")
        else:
            # Replace ".html" in the link with ".pdf"
            pdf_html_path = link.split(".html", 1)[0] + ".pdf"
            logger.debug(f"New URL after replacing .html: {pdf_html_path}")
            
        pdf_name = pdf_html_path.split("/")[-1]
        
        # Download PDFs only if they haven't already been downloaded
        if pdf_html_path not in unique_pdf_paths:
            unique_pdf_paths.append(pdf_html_path)

            try:
                # Attempt to load the PDF file from a local path and return the PDF as a Langchain Document object
                pdf_doc = download_pdf_and_return_doc(html_path=pdf_html_path,
                                                     pdf_filename=pdf_name,
                                                     proxies=proxies)

                if is_cisco_datasheet==True:
                    # Add metadata if PDF is a Cisco datasheet
                    pdf_doc = generate_cisco_metadata(pdf_doc)
                    unique_pdf_docs.append(pdf_doc)
                else:
                    unique_pdf_docs.append(pdf_doc)

            except Exception as e:
                logger.info(f"Error occurred: {link} not available as PDF file.")
                logger.exception(e)
    
    num_unique_pdfs = len(unique_pdf_docs)
    
    logging.info(f"{num_unique_pdfs} links actually contained PDF files.")
    
    return unique_pdf_docs