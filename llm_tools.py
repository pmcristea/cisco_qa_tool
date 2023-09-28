import logging

import streamlit as st

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(levelname)s:%(name)s:%(funcName)s:%(message)s")

file_handler = logging.FileHandler('llm_tools.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)



def get_llm_response(vectordb: langchain.vectorstores.pinecone.Pinecone, 
                     product_question: str,
                     top_k_chunks: int, 
                     model: str) -> dict:
    """Returns a dictionary containing information about a query to a LLM.  The dictionary contains the product_question itself, the LLM's response to the product_question, and the chunks that were pulled from vectordb to answer the question.

    Args:
        vectordb: a vectordatabase containing source documents used to answer product_question
        top_n_chunks: How many chunks are retrieved from the vector store to be used in answering the product_question
        model: Name of the LLM model to use
        product_question: A question you want answered about a video game
    
    Returns:
        llm_response: 
    """
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
    from langchain.chat_models import ChatOpenAI
    
    # Set search parameters to return the top_k_chunks from the vectordb
    search_kwargs = {'k': top_k_chunks}
    
    # Create retriever object which can search and retrieve chunks from the vectordb
    retriever = vectordb.as_retriever(search_kwargs=search_kwargs)
    

    # Write template containing prompts or instructions for the llm model to use in its response.
    prompt_template = """Use the following pieces of context to answer the question.  If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
{context}

Question: {question}
Answer:
"""
    # Set up prompt template
    PROMPT = PromptTemplate(template=prompt_template, 
                            input_variables=["context", "question"])
    chain_type_kwargs = {"prompt": PROMPT}
    
    # Create llm object based on model_name
    llm = ChatOpenAI(model_name=model)
    
    # Create RetrievalQA chain used to generate response to questions
    qa_chain = RetrievalQA.from_chain_type(
                                    llm=llm, 
                                    chain_type="stuff", 
                                    retriever=retriever, 
                                    return_source_documents=True,
                                    chain_type_kwargs=chain_type_kwargs
                                    )
    
    st.write(f"Searching knowledgebase for answer to {product_question}")
    
    # Generate response to product_question
    llm_response = qa_chain(product_question)
    
    return llm_response