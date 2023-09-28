import logging

import langchain

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
                               ) -> langchain.vectorstores.pinecone.Pinecone:
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
    from langchain.vectorstores import Pinecone
    from langchain.embeddings import OpenAIEmbeddings
    
    # intialize pinecone
    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"), # find at app.pinecone.io
        environment=os.getenv("PINECONE_ENV"), # next to api key in console
    )

    # Check if the index already exists.  If it doesn't, then create one 
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            name=index_name,
            metric='cosine',
            dimension=1536
        )
        
        logger.debug(f"New Pinecone Index called {index_name} created.")
        
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