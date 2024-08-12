from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from dotenv import load_dotenv
import streamlit as st
import os
from langchain_community.document_loaders import YoutubeLoader,WebBaseLoader, UnstructuredURLLoader
import logging
import validators
from langchain.prompts import PromptTemplate

log = logging.getLogger(__name__)

load_dotenv()
os.environ['Groq_API_KEY']=os.getenv('Groq_API_KEY')
#os.environ['USER_AGENT']=os.environ.get('USER_AGENT')

def get_user_agent() -> str:
    """Get user agent from environment variable."""
    env_user_agent = os.environ.get("USER_AGENT")
    if not env_user_agent:
        log.warning(
            "USER_AGENT environment variable not set, "
            "consider setting it to identify your requests."
        )
        return "DefaultLangchainUserAgent"
    return env_user_agent



groq_api_key=os.getenv("Groq_API_KEY")
llm=ChatGroq(model="Gemma-7b-It",api_key=groq_api_key)

prompt_template="""
Provide a Summary of the following content in 300 words :
Content:{text}
"""
prompt=PromptTemplate(template=prompt_template,input_variables=["text"])

st.set_page_config(page_title="Summarize Text from YT or Website",page_icon="/ai_icon.png")
st.title("Summarize Text from Youtube or Website")
st.subheader("Summarize URL")
inp_url=st.text_input("Enter Website Url/Youtube video link",label_visibility="collapsed")

with st.sidebar:
    st.text_input("Optional:Enter your own OpenAI Key",value="",type="password")

if st.button("Summarize the content from Youtube or Website"):
    if not inp_url:
        st.error("Please provide the Valid URL")
    elif not validators.url(inp_url):
        st.error("Please Try again with a Valid URL!")
    else:
        try:
            with st.spinner("Waiting..."):
                if "youtube.com" in inp_url:
                    loader=YoutubeLoader.from_youtube_url(inp_url,add_video_info=True)
                else:
                    loader=UnstructuredURLLoader(urls=[inp_url],ssl_verify=False,headers={"user-Agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 "})

                data=loader.load()

                chain=load_summarize_chain(llm=llm,chain_type='refine',verbose=True)
                text_summary=chain.run(data)
                st.success(text_summary)
    
        except Exception as e:
            st.exception(f"Exception:{e}")
