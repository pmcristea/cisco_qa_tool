from requests_html import HTMLResponse

import langchain

import streamlit as st


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
    
    import streamlit as st
    
    # Combine user input to construct the query used for the Google search
    search_args = [website_url, product_name, additional_search_keywords]
    joined_search_args = " ".join(search_args)
    query_suffix = urllib.parse.quote_plus(joined_search_args)  
    query = f"https://www.google.com/search?q=site%3A{query_suffix}&lr=lang_{language_code}"
    
    if verbose:
        st.write(query_suffix)
        print(query_suffix)
        st.write(query)
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
        st.write(f"{len(links)} links found initially.  Removing links that aren't useful.")
        print(f"{len(links)} links found initially.  Removing links that aren't useful.")
        st.write(links)
        print(links)
    
    # Remove search results based on the criteria established earlier
    for url in links[:]:
        if url.startswith(google_domains):
            links.remove(url)
        if url.endswith(suffixes):
            links.remove(url)
    
    # Sort remaining links
    links.sort()
    
    st.write(f"{len(links)} useful links found.")
    print(f"{len(links)} useful links found.")
    
    return links         
            
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
    
    import streamlit as st
       
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
            st.write(f"{temp_file_path} written to disk")
            print(f"{temp_file_path} written to disk")
            
        return pdf_doc


def generate_cisco_metadata(pdf_doc: list, verbose: bool = False) -> list:
    """Extracts information from the URL of a data sheet pulled from www.cisco.com to be used as metadata in the document.

    Args:
        pdf_doc (list): A list where each element is a single page of a pdf file
        verbose (boolean): Used for debugging

    Returns:
        new_pages (list): A list of pdf pages after Cisco metadata has been added
    """
    
    import streamlit as st
    
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
            pass
        
        new_pages.append(page)
            
    if verbose:
        if failed:
            st.write(f"Failed to get metadata for {pdf_html_path}")
            print(f"Failed to get metadata for {pdf_html_path}")
        else:
            st.write(f"""Product category: {pdf_doc[0].metadata["product_category"]}, \nProduct_name: {pdf_doc[0].metadata["product_name"]}, \nMetadata added for {pdf_doc[0].metadata["filename"]}.""")
            print(f"""Product category: {pdf_doc[0].metadata["product_category"]}, \nProduct_name: {pdf_doc[0].metadata["product_name"]}, \nMetadata added for {pdf_doc[0].metadata["filename"]}.""")
    
    return new_pages


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
    
    import streamlit as st

    unique_pdf_docs = []
    unique_pdf_names = []

    for link in links:
        # Cisco datasheets are available as both a HTML file and a PDF file.  Any datasheet that has a URL ending with .html can be accessed from the same URL if the .html is replaced with .pdf.  
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
                    
                st.write(f"File {pdf_name} downloaded and successfully loaded.")
                print(f"File {pdf_name} downloaded and successfully loaded.")

            except Exception as e:
                #print(e)
                if verbose:
                    st.write(f"Error occurred: {link} not available as PDF file.")
                    print(f"Error occurred: {link} not available as PDF file.")
            #     # Cleanup step removing PDF file after being loaded as it isn't needed anymore
            # if os.path.exists(pdf_local_path):
            #     os.remove(pdf_local_path)

    return unique_pdf_docs


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


def return_new_links(links: list, vectordb, verbose: bool = False) -> list:
    """Checks an existing vectordb whether it contains any documents having URLs matching those in links.  Any URLs found to match are removed as they imply the document already exists in the vectordb and doesn't need to be added again.

    Args:
        links (list): a list of URLs
        vectordb (): a vectordatabase 
        debug (bool): prints when a URL doesn't match any that already exist in the vectordb.
    
    Returns:
        new_links (list): a list of URLs not already present in vectordb.
    """
    
    import streamlit as st
    
    new_links = []

    for url in links:
        # Find which urls in links match with PDF files already in the vectordb
        pdf_html_path = url.split(".html", 1)[0] + ".pdf"
        matches = vectordb.similarity_search(query=' ', 
                                             k=1, 
                                             filter={'source':pdf_html_path})

        # If any matches are found, it implies the PDF file exists in the vectordb and we can ignore the url
        if len(matches)>0:
            pass
        
         # If no matches are found, keep the URL so the PDF file can be added to vectordb later
        else:
            if verbose==True:
                st.write(f'{url} not found in database')
                print(f'{url} not found in database')
            new_links.append(url)
    
    return new_links


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
    
    import streamlit as st
    
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
                st.write(f'{filename} not found in database')
                print(f'{filename} not found in database')
            new_docs.append(doc)
    st.write(f"{len(new_docs)} new document(s) not already present in the database.")
    print(f"{len(new_docs)} new document(s) not already present in the database.")
    
    return new_docs


def token_counter(text):
    import tiktoken
    
    tokenizer = tiktoken.get_encoding('cl100k_base')
    
    tokens = tokenizer.encode(text,
                              disallowed_special=()
                             )
    return len(tokens)


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
    
    import streamlit as st
    
    # Set max_tokens to be 10_000 less than the limit OpenAI enforces
    max_tokens = 1_000_000 - 10_000
    
    # Flatten a list of lists into a single list containing each page of all PDF docs in texts
    flattened_pages = [page for pdf_doc in texts for page in pdf_doc]
    
    # Total number of pages across all PDF docs
    num_pages = len(flattened_pages)
    st.write(f'{num_pages} pages of documents must be inserted into database')
    print(f'{num_pages} pages of documents must be inserted into database')
    
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
            st.write(f"Inserted {pages_inserted} pages into database.  {remaining_pages} pages remaining.")
            print(f"Inserted {pages_inserted} pages into database.  {remaining_pages} pages remaining.")
            vectordb.add_documents(batched_pages)
            # Reset the total_tokens count and which pages are in the batch so they're ready for the rest of the loop.  
            total_tokens = 0
            batched_pages = []
            
            # Sleep for 60 seconds to make sure the limit isn't hit on the next batch
            st.write('sleeping for 60 seconds')
            print('sleeping for 60 seconds')
            time.sleep(60)
    
    # If the last iteration of the for loop occurs and the if condition is true, then the batched_texts won't get uploaded.  This checks to see if any batched texts are present after the loop is complete and adds them. 
    if len(batched_pages)>0:
        pages_inserted=len(batched_pages)
        st.write(f"Inserted {pages_inserted} pages into database.  No pages remaining.")
        print(f"Inserted {pages_inserted} pages into database.  No pages remaining.")
        vectordb.add_documents(batched_pages)
    
    st.write("Process completed.")
    print("Process completed.")
    #vectordb.persist()
    
    
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
    
    import streamlit as st

    st.write('\n\nAnswer:')
    print('\n\nAnswer:')    
    st.write(llm_response['result'])
    print(llm_response['result'])
    
    if print_sources:
        st.write('\n\nSources:')
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
            st.write(f"{i}. {source_url}#page={source_page_num} - Page {source_page_num}")
            print(f"{i}. {source_url}#page={source_page_num} - Page {source_page_num}")
    st.write()
    print()
    
    if print_chunks:
        st.write('\n\nChunks:')
        print('\n\nChunks:')
        for i, chunk in enumerate(llm_response["source_documents"], 1):
            st.write(f'----------Chunk {i}----------')
            st.write(chunk.page_content)
            st.write()
            print(f'----------Chunk {i}----------')
            print(chunk.page_content)
            print()
            
    return None
            

@st.cache_data(ttl=600)
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
    
    import streamlit as st
    
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
        st.write(f'{num_new_links} new URLs detected')
        print(f'{num_new_links} new URLs detected')
        
        if verbose:
            st.write(new_links)
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
            st.write("No new docs to add")
            print("No new docs to add")
    else:
        st.write("No new links to add")
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