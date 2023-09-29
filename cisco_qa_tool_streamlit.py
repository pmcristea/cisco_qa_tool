import streamlit as st

from web_tools import get_source, scrape_google
from pdf_tools import load_pdf_as_doc, download_pdf_and_return_doc, generate_cisco_metadata, return_pdf_docs
from vectorstore_tools import return_pinecone_vectorstore, batch_and_add_texts
from return_new import return_new_links, return_new_docs
from llm_tools import get_llm_response, process_llm_response

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(levelname)s:%(name)s:%(funcName)s:%(message)s")

file_handler = logging.FileHandler('cisco_qa_tool_streamlit.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# PROXIES = {
#     "http":f"http://{st.secrets['PROXY_NAME']}:{st.secrets['PROXY_PASSWORD']}@{st.secrets['PROXY_URL']}:{st.secrets['PROXY_PORT']}",
#     "https":f"http://{st.secrets['PROXY_NAME']}:{st.secrets['PROXY_PASSWORD']}@{st.secrets['PROXY_URL']}:{st.secrets['PROXY_PORT']}"
# }

@st.cache_data(ttl=600)
def cisco_qa_search_tool(product_question: str, 
                         product_name: str,
                         index_name: str = "cisco-knowledgebase", 
                         top_n_chunks: int = 5, 
                         print_sources: bool = False, 
                         print_chunks: bool = False, 
                         model:str = 'gpt-4',
                         proxies: dict = None):
    """
    
    """

    
    links = scrape_google(website_url="www.cisco.com",
                          product_name=product_name, 
                          additional_search_text="data sheets",
                          PROXIES=proxies)
    
    vectordb = return_pinecone_vectorstore(index_name)
    
    new_links = return_new_links(links, 
                                 vectordb)

    num_new_links = len(new_links)
    
    if num_new_links > 0:       
        docs = return_pdf_docs(new_links,
                               proxies=proxies,
                               is_cisco_datasheet=True)
        
        new_docs = return_new_docs(docs, 
                                   vectordb)
        
        if len(new_docs) > 0:
            batch_and_add_texts(new_docs, vectordb)
            
        else:
            st.write("No new docs to add")
            logger.info("No new docs to add")
    else:
        st.write("No new links to add")
        logger.info("No new links to add")
        
    
    llm_response = get_llm_response(vectordb, 
                                    product_question,
                                    product_name,
                                    top_n_chunks, 
                                    model)
    
    processed_response = process_llm_response(llm_response, 
                                              print_sources,
                                              print_chunks)
    
    return processed_response


# Page title
st.set_page_config(page_title='🦜🔗 Data Sheet Answering Tool')
st.title('🦜🔗 Data Sheet Answering Tool')

# Query text
product_name_input = st.text_input('Enter the name of the product you have a question about:', placeholder = "What is the name of the product?")
query_input = st.text_input(f'Enter your question about the product:', placeholder = 'What do you want to ask about the product?')

result = []
with st.form('myform', clear_on_submit=True):
    submitted = st.form_submit_button('Submit', disabled=not(product_name_input and query_input))
    if submitted:
        with st.spinner(f"Please be patient, it may take some time to get your answer.  If no documents pertaining to your question are currently present in the knowledge base, they'll have to be added."):
            cisco_qa_search_tool(product_question=query_input,
                                 product_name=product_name_input,
                                 top_n_chunks = 10,
                                 print_sources=True,
                                 print_chunks=False,
                                 model='gpt-4', 
                                 proxies=None)