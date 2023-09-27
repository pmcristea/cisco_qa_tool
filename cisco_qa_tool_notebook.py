# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from requests_html import HTMLResponse

from dotenv import find_dotenv, load_dotenv

import os

import langchain

# %%
load_dotenv(find_dotenv("../.env"))

# %%
db_path = os.getenv('DB_PATH')
proxy_name = os.getenv('PROXY_NAME')
proxy_password = os.getenv('PROXY_PASSWORD')
proxy_url = os.getenv('PROXY_URL')
proxy_port = os.getenv('PROXY_PORT')

# %%
PROXIES = {"http":f"http://{proxy_name}:{proxy_password}@{proxy_url}:{proxy_port}",
           "https":f"http://{proxy_name}:{proxy_password}@{proxy_url}:{proxy_port}"}

# %%
website = 'www.cisco.com'
additional_keywords = 'data sheet'


# %% [markdown]
# # get_source

# %%
## Code found from https://practicaldatascience.co.uk/data-science/how-to-scrape-google-search-results-using-python, with minor adjustments by myself.

def get_source(url: str, PROXIES: dict = None) -> HTMLResponse:
    """Return the source code for the provided URL.
    
    Args:
        url (string): URL of the page to scrape.
        PROXIES (dict): dictionary of proxy IP addresses to use for HTTP requests
        
    Returns:
        response (object): HTTP response object.
    """  
    
    from fake_useragent import UserAgent
    
    import requests
    from requests.adapters import HTTPAdapter
    
    from requests_html import HTMLSession
    
    from urllib3.util.retry import Retry
    
    # Randomly generate headers to make HTTP get requests look like a user and not a bot
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
    
    return response


# %%
get_source("https://papir805.github.io")


# %% [markdown]
# # scrape_google

# %%
def scrape_google(website_url: str, 
                  product_name: str, 
                  additional_search_keywords: str = 'none',
                  language_code: str = 'en',
                  PROXIES: dict = None,
                  verbose: bool = False) -> list:
    """Return a list of URLs corresponding to a Google search focusing on a particular website.  Results from certain URLs that don't contain walkthroughs are excluded based on the URL's prefixes and suffixes.

    Args:
        website_url (string): URL for Google to focus its search on.  Example: www.yoururlhere.com
        product_name (string): Name of the product to search for information on.
        additional_search_keywords (string): Any additional text you want to include in the search to help refine the results.
        language_code (boolean): Specifies which language to return search results in.
        PROXIES (dict): proxy IP addresses to use for HTTP requests
        verbose (boolean): Used for debugging

    Returns:
        links (list): A list of URLs corresponding to most recommended walkthroughs from www.gamefaqs.com for video_game_title on video_game_system.
    """
    import urllib
    
    # Combine user input to construct the query used for the Google search
    search_args = [website_url, product_name, additional_search_keywords]
    joined_search_args = " ".join(search_args)
    query_suffix = urllib.parse.quote_plus(joined_search_args)  
    query = f"https://www.google.com/search?q=site%3A{query_suffix}&lr=lang_{language_code}"
    
    if verbose:
        print(query_suffix)
        print(query)
    
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
    suffixes = ()
    
    if verbose==True:
        print(f"{len(links)} links found initially.  Removing links that aren't useful.")
        print(links)
    
    # Remove search results based on the criteria established earlier
    for url in links[:]:
        if url.startswith(google_domains):
            links.remove(url)
        if url.endswith(suffixes):
            links.remove(url)
    
    # Sort remaining links
    links.sort()
    
    print(f"{len(links)} useful links found.")
    
    return links


# %%
product_name = 'catalyst 9400'

links1 = scrape_google(website_url=website, 
                       product_name=product_name,
                       additional_search_keywords=additional_keywords,
                       verbose=True)
links1

# %%
product_name = 'catalyst 2500 wireless controller'

links2 = scrape_google(website_url=website, 
                       product_name=product_name,
                       additional_search_keywords=additional_keywords)
links2

# %%
product_name = 'catalyst 2500 wireless controller'

links2_1 = scrape_google(website_url=website, 
                       product_name=product_name,
                       additional_search_keywords=additional_keywords)
links2_1

# %%
product_name = 'catalyst 4500e'

links3 = scrape_google(website_url=website, 
                       product_name=product_name,
                       additional_search_keywords=additional_keywords)
links3

# %%
product_name = 'ir829'

links4 = scrape_google(website_url=website, 
                       product_name=product_name,
                       additional_search_keywords=additional_keywords)
links4


# %% [markdown]
# # load_pdf_as_doc

# %%
def load_pdf_as_doc(html_path: str, local_path, filename: str) -> list[langchain.schema.document.Document]:
    """Loads PDF file from local_path and returns a list where each element is one page of the PDF file as a Langchain Document object.

    Args:
        html_path (string): URL where the PDF file was originally downloaded from
        local_path (string): File path used to read the PDF file from local disk

    Returns:
        pdf_doc (list[langchain.schema.document.Document]): A list where each element is one page of the PDF file as a Langchain Document object
    """
    from langchain.document_loaders import PyPDFLoader
    
    # Load local PDF file into memory 
    loader = PyPDFLoader(file_path=str(local_path))
    
    # Create a Langchain Document object for each page in the PDF file
    pdf_doc = loader.load()
    
    # Add the URL where the PDF was downloaded from as metadata
    for page in pdf_doc:
        page.metadata['source'] = html_path
        page.metadata['filename'] = filename
        page.metadata['page'] = int(page.metadata['page']) + 1
    
    return pdf_doc


# %% [markdown]
# # download_pdf_and_return_doc

# %%
def download_pdf_and_return_doc(html_path: str, 
                                 pdf_filename: str, 
                                 proxies=None, 
                                 verbose=False):
    """
    Initiates HTTP get request for the URL of a pdf file.  Writes the contents of the PDF file to local disk.

    Args:
        html_path (string): HTML URL of the PDF file
        local_path (string): File path for saving the PDF file to local disk
        proxies (dict): proxy IP addresses to use for HTTP requests
        verbose (boolean): Used for debugging

    Returns:
        new_pages (list): A list of pdf pages after Cisco metadata has been added
    """
    
    import tempfile
    import pathlib
       
    temp_dir = tempfile.TemporaryDirectory()
    temp_file_path = pathlib.Path(temp_dir.name) / pdf_filename
    
    # Get HTTP response of a URL
    response = get_source(html_path, proxies)
    
    # Check to see if the HTTP response is actually for a PDF file
    if response.headers['Content-Type'] == 'application/pdf':
        # If the file is a PDF, write the contents to local_path            
        with open(temp_file_path, 'wb') as pdf:
            pdf.write(response.content)
            pdf_doc = load_pdf_as_doc(html_path, temp_file_path, pdf_filename)
            
            
    
        if verbose:
            print(f"{temp_file_path} written to disk")
            
        return pdf_doc

# %%
weird_response = get_source("https://www.cisco.com/en/US/prod/collateral/routers/ps12558/ps12559/datasheet-c78-730862.pdf")

# %%
weird_response.headers['Content-Type']

# %%
pdf_doc = download_pdf_and_return_doc("https://www.cisco.com/en/US/prod/collateral/routers/ps12558/ps12559/datasheet-c78-730862.pdf", f"datasheet-c78-730862.pdf", verbose=True)

# %% jupyter={"outputs_hidden": true}
generate_cisco_metadata(pdf_doc)

# %% jupyter={"outputs_hidden": true}
pdf_doc[0]


# %% [markdown]
# # generate_cisco_metadata

# %%
def generate_cisco_metadata(pdf_doc: list, verbose: bool = False) -> list:
    """Extracts information from the URL of a data sheet pulled from www.cisco.com to be used as metadata in the document.

    Args:
        pdf_doc (list): A list where each element is a single page of a pdf file
        verbose (boolean): Used for debugging

    Returns:
        new_pages (list): A list of pdf pages after Cisco metadata has been added
    """
    
    new_pages = []
    failed_docs = []
    failed = False
    
    for page in pdf_doc:  
    # Many source URLs for Cisco datasheets follow this structure:
    # https://www.cisco.com/.../collateral/product_category/product_name/filename.pdf

    # Separates the URL into a list of the form:
    # ['...', 'www.cisco.com', '...', 'collateral', 'product_category', 'product_name', 'filename']
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
            # If "collateral" isn't present in the split_url, don't generate any new metadata
            failed = True
            print(e)
            pass
        
        new_pages.append(page)
            
        if verbose:
            if failed:
                print(f"Failed to get metadata for {pdf_html_path}")
            else:
                print(f"""Product category: {pdf_doc[0].metadata["product_category"]}, \nProduct_name: {pdf_doc[0].metadata["product_name"]}, \nMetadata added for {pdf_doc[0].metadata["filename"]}.""")
    
    return new_pages


# %%
test_list = ['a', 'collateral', 'b']
test_list.index('z')

# %%
new_doc = generate_cisco_metadata(pdf_doc, verbose=True)

# %%
new_doc[0]


# %% [markdown]
# # return_pdf_docs

# %%
def return_pdf_docs(links: list, 
                    verbose: bool = False, 
                    proxies: dict = None,
                    is_cisco_datasheet: bool = False) -> list:
    """
    Download PDFs from URLs, load from local disk as Langchain Documents and add metadata.  Filters URLs so that duplicate PDF files are ignored.

    Args:
        links (list): A list of URLs
        verbose (boolean): Used for debugging
        proxies (dict): proxy IP addresses to use for HTTP requests
        is_cisco_datasheet (boolean): Extracts information to be used as metadata based on criteria specifically designed for Cisco data sheets

    Returns:
        unique_pdf_docs (list): A list of Langchain PDF documents
    """

    unique_pdf_docs = []
    unique_pdf_names = []

    for link in links:
        # Cisco datasheets are available as both a HTML file and a PDF file.  Any datasheet that has a URL ending with .html can be accessed from the same URL if the .html is replaced with .pdf.  
        if link.endswith(".pdf"):
            pdf_html_path = link
        else:
            pdf_html_path = link.split(".html", 1)[0] + ".pdf"
        
        pdf_name = pdf_html_path.split("/")[-1]
        #pdf_local_path = f"./data/{pdf_name}"
        
        if pdf_name not in unique_pdf_names:
            # Download PDFs only if they haven't already been downloaded
            unique_pdf_names.append(pdf_name)

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

                print(f"File {pdf_name} downloaded and successfully loaded.")

            except Exception as e:
                print(f"Error occurred: {link} not available as PDF file.")
                if verbose:
                    print(e)
                    print(f"Error occurred: {link} not available as PDF file.")
            #     # Cleanup step removing PDF file after being loaded as it isn't needed anymore
            # if os.path.exists(pdf_local_path):
            #     os.remove(pdf_local_path)

    return unique_pdf_docs


# %%
pdf_docs1 = return_pdf_docs(links1, is_cisco_datasheet=True)

# %%
for doc in pdf_docs1:
    print(doc[0].metadata)

# %%
pdf_docs2 = return_pdf_docs(links2, is_cisco_datasheet=True)

# %%
for doc in pdf_docs2:
    print(doc[0].metadata)

# %%
pdf_docs3 = return_pdf_docs(links3, is_cisco_datasheet=True, verbose=True)

# %%
for doc in pdf_docs3:
    print(doc[0].metadata)

# %%
pdf_docs4 = return_pdf_docs(links4, is_cisco_datasheet=True)

# %%
for doc in pdf_docs4:
    print(doc[0].metadata)


# %% [markdown]
# # return_pinecone_vectorstore

# %%
def return_pinecone_vectorstore(index_name: str, 
                                model_name: str = 'text-embedding-ada-002'
                               ) -> langchain.vectorstores.pinecone.Pinecone:
    """
    
    """
        
    import pinecone
    import os
    from langchain.vectorstores import Pinecone
    from langchain.embeddings import OpenAIEmbeddings
    
    # intialize pinecone
    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"), # find at app.pinecone.io
        environment=os.getenv("PINECONE_ENV"), # next to api key in console
    )

    # First, check if our index already exists.  If it doesn't we create it
    if index_name not in pinecone.list_indexes():
        # we create a new index
        pinecone.create_index(
            name=index_name,
            metric='cosine',
            dimension=1536
        )
        
    # Specify which vector embeddings to use
    embedding = OpenAIEmbeddings(model=model_name)
    
    # Create the Langchain vector database object
    index = pinecone.Index(index_name)
    vectordb = Pinecone(index, embedding, 'text')
    
    return vectordb


# %%
index_name = "cisco-knowledgebase"
vectordb_pinecone = return_pinecone_vectorstore(index_name)


# %% [markdown]
# # return_chroma_vectorstore

# %%
def return_chroma_vectorstore(persist_directory: str) -> langchain.vectorstores.chroma.Chroma:
    """
    If a persist_directory is provided, attemps to return the vector store located there.  If a persist_directory is not provided, one is created called 'db' by default.
    
    Returns:
        vectordb (langchain.vectorstores.chroma.Chroma): A langchain object that is used to access the vectorstore called 'db' that was created.
    """
    
    from langchain.vectorstores import Chroma
    from langchain.embeddings import OpenAIEmbeddings
    
    # Specify which vector embeddings to use
    embedding = OpenAIEmbeddings()
    
    if persist_directory:
        # Get vectorstore from persist_directory on local disk
        print(f'getting vectorstore named {persist_directory}')
        vectordb = Chroma(persist_directory=persist_directory,
                      embedding_function=embedding)
        print('got vectorstore')
    else:
        # Create vectorstore at ./db on local disk
        print(f"persist directory not given \ncreating vectorstore called db")
        vectordb = Chroma(persist_directory='db',
                      embedding_function=embedding)
        print('vectorstore created')       
        vectordb.persist()
    
    return vectordb


# %%
vectordb_chroma = return_chroma_vectorstore(f"{db_path}/cisco_db/")

# %% [markdown]
# # return_new_links

# %%
web_url = "https://www.cisco.com/c/dam/global/en_sg/training-events/datacentertechbyte/assets/pdfs/n1000_datasheet.pdf"

web_url.split(".html", 1)[0] + ".pdf"


# %%
def return_new_links(links: list, vectordb, verbose: bool = False) -> list:
    """Checks an existing vectordb whether it contains any documents having URLs matching those in links.  Any URLs found to match are removed as they imply the document already exists in the vectordb and doesn't need to be added again.

    Args:
        links (list): a list of URLs
        vectordb (): a vectordatabase 
        debug (bool): prints when a URL doesn't match any that already exist in the vectordb.
    
    Returns:
        new_links (list): a list of URLs not already present in vectordb.
    """
    
    new_links = []

    for url in links:
        if url.endswith('.pdf'):
            pdf_html_path = url
        # Find which urls in links match with PDF files already in the vectordb
        else:
            pdf_html_path = url.split(".html", 1)[0] + ".pdf"
        
        matches = vectordb.similarity_search(query=' ', 
                                             k=1, 
                                             filter={'source':pdf_html_path})
        # If any matches are found, it implies the PDF file exists in the vectordb and we can ignore the url
        if len(matches)>0:
            continue
        
         # If no matches are found, keep the URL so the PDF file can be added to vectordb later
        else:
            if verbose==True:
                print(f'{url} not found in database')
            new_links.append(url)
    
    return new_links


# %%
new_links = return_new_links(links1, vectordb_pinecone, verbose=True)


# %% [markdown]
# # return_new_docs

# %%
def return_new_docs(pdf_docs: list, 
                    vectordb, 
                    verbose: bool = False) -> list:
    """Checks an existing vectordb whether it contains any documents having a filename matching one already in the vectordb.  Any documents found to match are ignored as they imply the document already exists in the vectordb and doesn't need to be added again.

    Args:
        docs (list): a list of LangChain Document objects
        vectordb (): a vectordatabase
        debug (bool): prints when a document for a particular video_game_title and walkthrough_id doesn't match any already present in the vectordb
    
    Returns:
        new_docs (list): a list of LangChain Document objects not already present in vectordb.
    """
    
    new_docs = []
    
    # Check if any pdf_docs have a filename that already exist in vectordb.
    for doc in pdf_docs:
        filename = doc[0].metadata['filename']
        matches =  vectordb.similarity_search(query=' ',k=1,filter=
                                        {
                                            "filename": {"$eq": filename}
                                        }
                                             )
        # If any matches are present, the doc already exists in the vectordb and can be ignored 
        if len(matches) > 0:
            pass
        
        # If no matches are found, keep the doc so it can be added to vectordb later
        else:
            if verbose:
                print(f'{filename} not found in database')
            new_docs.append(doc)
    print(f"{len(new_docs)} new document(s) not found in database.")
    
    return new_docs


# %%
new_docs1 = return_new_docs(pdf_docs1, vectordb_pinecone, verbose=True)

# %%
new_docs2 = return_new_docs(pdf_docs2, vectordb_pinecone, verbose=True)

# %%
new_docs3 = return_new_docs(pdf_docs3, vectordb_pinecone, verbose=True)

# %%
new_docs4 = return_new_docs(pdf_docs4, vectordb_pinecone)

# %%
new_docs4[0][0].metadata


# %% [markdown]
# # token_counter

# %%
def token_counter(text):
    import tiktoken
    
    tokenizer = tiktoken.get_encoding('cl100k_base')
    
    tokens = tokenizer.encode(text,
                              disallowed_special=()
                             )
    return len(tokens)


# %% [markdown]
# # batch_and_add_texts

# %%
def batch_and_add_texts(texts: list, vectordb) -> None:
    """
    OpenAI enforces a token/min limit for embedding tokens. This function avoids hitting that limit by splitings texts into batches less than or equal to the token_limit.  After each batch embedded and added to the vectorstore, the program waits 60 seconds to avoid hitting the limit.

    Args:
        texts (list): a list of chunked LangChain Document objects
        vectordb (): a vectordatabase
    
    Returns:
        None
    """
    
    import time
    
    # Set max_tokens to be 10_000 less than the limit OpenAI enforces
    max_tokens = 1_000_000 - 10_000
    
    # Flatten a list of lists into a single list containing each page of all PDF docs in texts
    flattened_pages = [page for pdf_doc in texts for page in pdf_doc]
    
    # Total number of pages across all PDF docs
    num_pages = len(flattened_pages)
    print(f'num_pages: {num_pages}')
    
    total_tokens = 0
    batched_pages = []
        
    for _ in range(num_pages):
        # Remove page from list, count how many tokens it has and add the count to total_tokens
        current_page = flattened_pages.pop(0)
        batched_pages.append(current_page)
        num_tokens = token_counter(current_page.page_content)
        total_tokens += num_tokens
        
        # If total_tokens is less than the limit, continue the loop and add more pages of the PDF file to the batch
        if total_tokens <= max_tokens:
            continue
        
        # If the max_tokens limit is exceeded, insert the batched pages into the vectordb.
        else:
            pages_inserted=len(batched_pages)
            remaining_pages=len(flattened_pages)
            print(f"Inserted {pages_inserted} pages into database.  {remaining_pages} pages remaining.")
            vectordb.add_documents(batched_pages)
            # Reset the total_tokens count and which pages are in the batch so they're ready for the rest of the loop.  
            total_tokens = 0
            batched_pages = []
            
            # Sleep for 60 seconds to make sure the limit isn't hit on the next batch
            print('sleeping for 60 seconds')
            time.sleep(60)
    
    # If the last iteration of the for loop occurs and the if condition is true, then the batched_texts won't get uploaded.  This checks to see if any batched texts are present after the loop is complete and adds them. 
    if len(batched_pages)>0:
        pages_inserted=len(batched_pages)
        print(f"Inserted {pages_inserted} pages into database.  Process completed.")
        vectordb.add_documents(batched_pages)
        
    #vectordb.persist()


# %%
batch_and_add_texts(pdf_docs1, vectordb_pinecone)


# %% [markdown]
# # get_llm_response

# %%
def get_llm_response(vectordb, 
                     product_question: str,
                     top_n_chunks: int, 
                     model: str) -> dict:
    """Returns a dictionary containing information about a query to a LLM.  The dictionary contains the product_question itself, the LLM's response to the product_question, and the chunks that were pulled from vectordb to answer the question.

    Args:
        vectordb (): a vectordatabase containing source documents used to answer product_question
        top_n_chunks (int): How many chunks are retrieved from the vector store to be used in answering the product_question
        model (str): Name of the LLM model to use
        product_question (str): A question you want answered about a video game
        video_game_titles (list): A list of titles corresponding to the video game being asked about.  Many video games on GameFAQs have different names corresponding to the same game.  This list will enable the RetrievalQA chain to focus attention on only documents that correspond to the game being asked about.
    
    Returns:
        llm_response (dict): 
    """
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
    from langchain.chat_models import ChatOpenAI
    
    search_kwargs = {'k': top_n_chunks}
    
    retriever = vectordb.as_retriever(search_kwargs=search_kwargs)
    

    
    prompt_template = """Use the following pieces of context to answer the question.  If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
{context}

Question: {question}
Answer:
"""
    
    PROMPT = PromptTemplate(template=prompt_template, 
                            input_variables=["context", "question"])
    
    chain_type_kwargs = {"prompt": PROMPT}
    llm = ChatOpenAI(model_name=model)

    qa_chain = RetrievalQA.from_chain_type(
                                    llm=llm, 
                                    chain_type="stuff", 
                                    retriever=retriever, 
                                    return_source_documents=True,
                                    chain_type_kwargs=chain_type_kwargs
                                    )
    
    llm_response = qa_chain(product_question)
    
    return llm_response


# %%
# llm_response = get_llm_response(vectordb_pinecone, query1, 10, model='gpt-3.5-turbo-16k')

# %% [markdown]
# # process_llm_response

# %%
def process_llm_response(llm_response: dict, 
                         print_sources: bool = False, 
                         print_chunks: bool = False) -> None:
    """Access and print the answer to a llm query.  Additionally print the URL of any source document used to answer the question, as well as which chunks from that source document were used.

    Args:
        llm_response (dict): A dictionary containing the response to a llm query and which source documents and chunks were used.
        print_sources (bool): Specifies whether to print the source documents used.
        print_chunks (bool): Specifies whether to print the chunks that were used.

    Returns:
        None
    """
    print(llm_response['result'])
    if print_sources:
        print('\n\nSources:')
        
        unique_sources = []

        for source in llm_response['source_documents']:
            source_url = source.metadata['source']
            source_page = source.metadata['page']
            sources = (source_url, source_page)
            if sources not in unique_sources:
                unique_sources.append(sources)
        
        for i, source in enumerate(unique_sources, 1):
            source_url = source[0]
            source_page_num = int(source[1])
            print(f"{i}. {source_url}#page={source_page_num} - Page {source_page_num}")
    print()
    if print_chunks:
        print('\n\nChunks:')
        for i, chunk in enumerate(llm_response["source_documents"], 1):
            print(f'----------Chunk {i}----------')
            print(chunk.page_content)
            print()


# %% [markdown]
# # cisco_qa_search_tool

# %%
def cisco_qa_search_tool(product_question: str, 
                         product_name: str,
                         index_name: str = "cisco-knowledgebase", 
                         top_n_chunks: int = 5, 
                         print_sources: bool = False, 
                         print_chunks: bool = False, 
                         model:str = 'gpt-3.5-turbo',
                         proxies: dict = None,
                         verbose: bool = False):
    """
    
    """
    
    links = scrape_google(website_url="www.cisco.com",
                          product_name=product_name, 
                          additional_search_keywords="data sheets",
                          PROXIES=proxies,
                          verbose=verbose)
    
    vectordb = return_pinecone_vectorstore(index_name)
    
    new_links = return_new_links(links, 
                                 vectordb, 
                                 verbose=verbose)

    num_new_links = len(new_links)
    
    if num_new_links > 0:
        print(f'{num_new_links} new URLs detected')
        
        if verbose:
            print(new_links)
        
        docs = return_pdf_docs(new_links, 
                               verbose=verbose, 
                               proxies=proxies,
                               is_cisco_datasheet=True)
        
        new_docs = return_new_docs(docs, 
                                   vectordb, 
                                   verbose=verbose)
        
        if len(new_docs) > 0:
            batch_and_add_texts(new_docs, vectordb)
            
        else:
            print("No new docs to add")
    else:
        print("No new links to add")
    
    question_with_product_name = product_question + ' ' + product_name
    
    llm_response = get_llm_response(vectordb, 
                                    question_with_product_name,
                                    top_n_chunks, 
                                    model)
    
    processed_response = process_llm_response(llm_response, 
                                              print_sources,
                                              print_chunks)
    
    return processed_response


# %% [markdown]
# # catalyst 2500 wireless controller

# %%
product_name = 'catalyst 2500 wireless controller'

# %% [markdown]
# ## query 1

# %%
query1 = 'how many access points do the cisco 2500 series wireless controllers have?'

# %%
cisco_qa_search_tool(product_question=query1,
                     product_name=product_name,
                     top_n_chunks = 5, 
                     print_sources=True,
                     print_chunks=True, 
                     model='gpt-3.5-turbo')

# %% [markdown]
# # catalyst 9400

# %%
product_name = 'catalyst 9400'

# %% [markdown]
# ## query1

# %%
query1 = "What ranges of temperatures can it endure?"

# %%
cisco_qa_search_tool(product_question=query1,
                     product_name=product_name,
                     top_n_chunks = 5, 
                     print_sources=True,
                     print_chunks=True, 
                     model='gpt-3.5-turbo')

# %% [markdown]
# # catalyst 4500e

# %%
product_name = 'catalyst 4500e'

# %% [markdown]
# ## query1

# %%
query1 = "What throughput does it have?"

# %%
cisco_qa_search_tool(product_question=query1,
                     product_name=product_name,
                     top_n_chunks = 10, 
                     print_sources=True,
                     print_chunks=True, 
                     model='gpt-3.5-turbo-16k')

# %% [markdown]
# # network analysis module

# %%
product_name = 'network analysis module'

# %% [markdown]
# ## query1

# %%
query1 = "what are the features of a network analysis module"

# %%
cisco_qa_search_tool(product_question=query1,
                     product_name=product_name,
                     top_n_chunks = 10, 
                     print_sources=True,
                     print_chunks=True, 
                     model='gpt-3.5-turbo-16k')

# %%
