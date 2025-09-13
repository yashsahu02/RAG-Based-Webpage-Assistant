import validators
import streamlit as st 
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough  # New import
from langchain_core.runnables import RunnableLambda 

## setup streamlit app 
st.set_page_config(
    page_title="LangChain: Summarize Text from Webstie",
    page_icon="🦜"
)

st.title("🦜 LangChain: Summarize Text Website")
st.subheader("Summarize URL")

## get the groq api key 
with st.sidebar:
    groq_api_key=st.text_input("Groq API Key", type="password", value="")
    
generic_url = st.text_input("URL", label_visibility="collapsed")

## Gemma Open Source Model of Google
if groq_api_key:
    llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key)

## define the prompt 
prompt_template = """
    Provide a summary of the following content in 300 words:
    Content: {text}
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

parser = StrOutputParser()

if st.button("Summarize"):
    ## validate all the input likes api or url etc.
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please Provide the required information")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL. It can may any website URL")
        
    else:
        try:
            with st.spinner("Processing.."):
                ## loading the website or yt video data 

                loader = UnstructuredURLLoader(urls=[generic_url], ssl_verify=False, 
                                               headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                    
                docs=loader.load()
                
                # chain for summarization 
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(docs)
                st.success(output_summary)
                
                
                    
        except Exception as e:
            st.exception(f"Exception Occured:{e}")