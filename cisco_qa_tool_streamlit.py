import streamlit as st

from cisco_helper_funcs import *

# PROXIES = {
#     "http":f"http://{st.secrets['PROXY_NAME']}:{st.secrets['PROXY_PASSWORD']}@{st.secrets['PROXY_URL']}:{st.secrets['PROXY_PORT']}",
#     "https":f"http://{st.secrets['PROXY_NAME']}:{st.secrets['PROXY_PASSWORD']}@{st.secrets['PROXY_URL']}:{st.secrets['PROXY_PORT']}"
# }

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
                                 model='gpt-3.5-turbo-16k', 
                                 proxies=None,
                                 verbose=True)