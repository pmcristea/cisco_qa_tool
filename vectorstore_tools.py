import logging

from langchain.vectorstores.pinecone import Pinecone

import streamlit as st

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(levelname)s:%(name)s:%(funcName)s:%(message)s")

file_handler = logging.FileHandler('vectorstore_tools.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

def return_pinecone_vectorstore(index_name: str, 
                                model_name: str = 'text-embedding-ada-002'
                               ) -> Pinecone:
    """
    Establishes a connection to a Pinecone vectorstore 

    Args:
        index_name: Name of the index for the vectorstore
        model_name: Name of vector embeddings to use for the vectorstore
        filename: filename of PDF 

    Returns:
        vectordb: A Langchain Pinecone vectorstore object
    """
        
    import pinecone
    import os
    import time
    from langchain.vectorstores import Pinecone
    from langchain.embeddings import OpenAIEmbeddings
    
    # intialize pinecone
    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"), # find at app.pinecone.io
        environment=os.getenv("PINECONE_ENV"), # next to api key in console
    )

    # Check if the index already exists.  If it doesn't, then create one 
    if index_name not in pinecone.list_indexes():
        st.write(f"Creating Pinecone Index called {index_name}.  Please be patient as this takes 90 seconds..") 
        logger.debug(f"Creating Pinecone Index called {index_name}.  Please be patient as this takes 90 seconds..")   
        pinecone.create_index(
            name=index_name,
            metric='cosine',
            dimension=1536
        )
               
        time.sleep(30)
        
    # Specify which vector embedding model to use
    embedding = OpenAIEmbeddings(model=model_name)
    
    st.write("Accessing knowledgebase")
    
    # Create the Langchain vector database object
    index = pinecone.Index(index_name)
    vectordb = Pinecone(index, embedding, 'text')
    logger.info("Connection established with Pinecone vectorstore.")

    return vectordb


def token_counter(text: str) -> int:
    """
    Counts the number of tokens in a string

    Args:
        text: the string to count the number of tokens from

    Returns:
        num_tokens: number of tokens present in text
    """
    
    import tiktoken
    
    tokenizer = tiktoken.get_encoding('cl100k_base')
    
    tokens = tokenizer.encode(text,
                              disallowed_special=()
                             )
    num_tokens = len(tokens)
    
    return num_tokens


def batch_and_add_texts(texts: list,
                        vectordb: Pinecone) -> None:
    """
    OpenAI enforces a token/min limit for embedding tokens. This function avoids hitting that limit by splitings texts into batches less than or equal to the token_limit.  After each batch is embedded and added to the vectorstore, the program waits 60 seconds to avoid hitting the limit.

    Args:
        texts: a list of chunked LangChain Document objects to add to vectordb
        vectordb: a vectordatabase
    
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
    logger.info(f'{num_pages} pages of documents must be inserted into database')
    
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
            logging.info(f"Inserted {pages_inserted} pages into database.  {remaining_pages} pages remaining.")
            vectordb.add_documents(batched_pages)
            # Reset the total_tokens count and which pages are in the batch so they're ready for the rest of the loop.  
            total_tokens = 0
            batched_pages = []
            
            # Sleep for 60 seconds to make sure the limit isn't hit on the next batch
            st.write(f'Rate limit: {max_tokens} per min is close to being reached.  Sleeping for 60 seconds to avoid hitting the limit.  Sorry for the delay.')
            time.sleep(60)
    
    # If the last iteration of the for loop occurs and the if condition is true, then the batched_texts won't get uploaded.  This checks to see if any batched texts are present after the for loop is complete and adds them. 
    if len(batched_pages)>0:
        pages_inserted=len(batched_pages)
        logger.info(f"Inserted {pages_inserted} pages into database.  No pages remaining.")
        vectordb.add_documents(batched_pages)
    
    logger.info("Process completed.  All pages have been batched and inserted into knowledgebase.")