import streamlit as st
import pandas as pd
import langchain
import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI
from pandasai import SmartDataframe
from langchain.agents import create_openapi_agent

#loading OpenAI API key
load_dotenv(dotenv_path=r"C:\Users\mrhodes\OneDrive - Deloitte (O365D)\Documents\Python\PersonalProjects\Streamlit_excel_tool\OPEN_AI_API_KEY.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["PANDASAI_API_KEY"] = OPENAI_API_KEY

def create_agent(df):
    model = ChatOpenAI(temperature=0.7, api_key=OPENAI_API_KEY)
    agent = SmartDataframe(df=df, config={"llm":model})
    return agent

def query_agent(query, agent):
    question = """
    You are a helpful agent that will help the user interact with a dataset. Please respond only in JSON format.
    Please answer the query down below:

    """ + query
    response = agent.chat(question)
    return response


st.title("Marshall's Magic GenAI Machine (Beta)")

#Sidebar - where file will be uploaded and possibly so additional documentation
#additional features - need to accept more than just .csv
with st.sidebar:
    uploaded_file = st.file_uploader('Please enter .csv file here and watch the magic happen.')

#Chat and dataframe will only appear once the file has been uploaded
#Loading data to dataframe and presenting to user
if uploaded_file is not None:
    with st.expander("Your Data:"):
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())
        agent = create_agent(df)

    with st.form("my_form"):
        text = st.text_area("What would you like to ask the magic AI Assistant:")
        submitted = st.form_submit_button("Submit")
        if submitted:
            response = query_agent(query=text, agent=agent)
            st.info(response)

    #Building the chat function
    #if "openai_model" not in st.session_state:
        #st.session_state["openai_model"] = "gpt-3.5-turbo"

    #Initialize chat history
    #if "messages" not in st.session_state:
        #st.session_state.messages = []
    
    #Display chat messages from history on app rerun
    #for message in st.session_state.messages:
        #with st.chat_message(message["role"]):
            #st.markdown(message["content"])

    #if prompt := st.chat_input("What would you like to ask the magic AI Assistant?"):
        #with st.chat_message("user"):
            #st.markdown(prompt)
        
        #st.session_state.messages.append({"role": "user", "content": prompt})

        #response = f"User message: {prompt}"
        #with st.chat_message("Magic AI Assitant"):
            #st.markdown(response)

        #st.session_state.messages.append({"role": "assistant", "content": response})