import logging

import langchain

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
                     product_name: str,
                     top_k_chunks: int, 
                     model: str) -> dict:
    """Returns a dictionary containing information about a query to a LLM.  The dictionary contains the product_question itself, the LLM's response to the product_question, and the chunks that were pulled from vectordb to answer the question.

    Args:
        vectordb: a vectordatabase containing source documents used to answer product_question
        product_question: A question you want answered about a video game
        product_name: The name of the product being asked about
        top_n_chunks: How many chunks are retrieved from the vector store to be used in answering the product_question
        model: Name of the LLM model to use

    
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
    
    question_with_product_name = product_question + ' ' + product_name
    st.write(f"Searching knowledgebase for answer to:\n {product_question}\n\nFor product: {product_name}")
    
    # Generate response to product_question
    llm_response = qa_chain(question_with_product_name)
    
    return llm_response



def process_llm_response(llm_response: dict, 
                         print_sources: bool = False, 
                         print_chunks: bool = False) -> None:
    """Prints the answer to a llm query.  Additionally print the URL of any source document used to answer the question, as well as which chunks from that source document were used.

    Args:
        llm_response: Containing the response to a llm query and which source documents and chunks were used.
        print_sources: Specifies whether to print the source documents in llm_response.
        print_chunks: Specifies whether to print the chunks in llm_response.

    Returns:
        None
    """

    st.write('\n\nAnswer:')
    logger.info('\n\nAnswer:')    
    st.write(llm_response['result'])
    logger.info(llm_response['result'])
    
    if print_sources:
        st.write('\n\nSources:')
        logger.info('\n\nSources:')
        
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
            logger.info(f"{i}. {source_url}#page={source_page_num} - Page {source_page_num}")
    st.write()
    
    if print_chunks:
        st.write('\n\nChunks:')
        logger.info('\n\nChunks:')
        for i, chunk in enumerate(llm_response["source_documents"], 1):
            st.write(f'----------Chunk {i}----------')
            st.write(chunk.page_content)
            st.write()
            logger.info(f'----------Chunk {i}----------')
            logger.info(chunk.page_content)
            logger.info()
            
    return None