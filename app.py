import streamlit as st
import pandas as pd
from pandasai import Agent
from pandasai.llm.openai import OpenAI
from pandasai import SmartDatalake, SmartDataframe
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from pandasai.responses.streamlit_response import StreamlitResponse
import matplotlib.pyplot as plt
import os
import statsmodels.api
from scipy import stats

## Don't initialize session_state variables here
## should initialize them after API key
# st.session_state.openai_key = None
# st.session_state.dl = []

st.set_page_config(page_title="GenAI for Structured Data",
                   layout="wide",
                   page_icon=":books::parrot:")

st.title("Chat with Your Data (NHG)")

st.sidebar.markdown(
    """
    ### Instructions:
    1. Click 'Load Data' to upload data:

    The AI agent has connected to 2 default datasets for testing purposes.

    2. Ask Your Question:    

    Type your question into the input box to get insights from the data.

    3. Handling Errors:

    If you receive an error messages, try rephrasing your question and submit it again.
    If the error persists, reload the webpage and try again. 

    ### Sample questions to help you start:
    1. calculate total number of visits by hospital
    2. calculate average patient age at visit by hospital
    3. bar plot number of total visit by visit type with hospital as color side by side 
    4. line plot average LOS by visit day of week by gender
    5. calculate the correlation between LOS and patient age at visit and the p-value 
    6. do a chi-square test between actual risk of readmission and visit hospital and show me the detailed results in a table 
    7. do a logistic regression with At_High_Risk_Readmission as y; visit type, visit day of week, age at visit as Xs, show me all B coefficients, odds ration and their confidence interval and p value
    8. build a prediction model using logistic regression algorithm with At_High_Risk_Readmission as y; visit type, visit day of week, age at visit as Xs, add the predicted risk of readmission for each visit, and then show the top 10 rows
    
    """
)

# if "prompt_history" not in st.session_state:
#     st.session_state.prompt_history = []
#
# if "AI_response_history" not in st.session_state:
#     st.session_state["AI_response_history"] = []

# Create a container to display the chatbot's responses
# stream_handler = StreamHandler(st.empty())

# if "langchain_messages" not in st.session_state:
#     st.session_state["langchain_messages"] = []

# Set up memory
msgs = StreamlitChatMessageHistory(key="langchain_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you today?")

# set up "dl"
if "dl" not in st.session_state:
    st.session_state["dl"] = []

with st.sidebar:
    st.session_state.openai_key = st.secrets["general"]["OPENAI_API_KEY"]
    # if st.session_state.get("openai_key") is None:
    #     st.session_state.uploaded_files = []
    #     #st.session_state.prompt_history = []
    #     #st.session_state.AI_response_history = []
    #     st.session_state.df = None
    #     st.session_state.dl = []
    #
    #     with st.form("API key"):
    #         key = st.text_input("OpenAI API Key", value="", type="password")
    #         if st.form_submit_button("Submit"):
    #             st.session_state.openai_key = key

    # st.session_state.uploaded_files = st.file_uploader(
    #     "Choose a file (CSV or Excel) in long format (one data point per row).",
    #     type=["csv", "xlsx"],
    #     accept_multiple_files=True
    # )

    if st.button("Load Data"):
        df1 = pd.read_csv("data1.csv")
        df2 = pd.read_csv("data2.csv")
        df = pd.DataFrame(df1)
        st.write(df.head())
        st.session_state.dl.append(df)
        df = pd.DataFrame(df2)
        st.write(df.head())
        st.session_state.dl.append(df)
        st.success("2 default datasets loaded successfully!")
        # for file in st.session_state.uploaded_files:
        #     if file.name.endswith(".xlsx"):
        #         df = pd.read_excel(file)
        #     elif file.name.endswith(".csv"):
        #         df = pd.read_csv(file)
        #     else:
        #         raise ValueError("File type not supported!")
        #     st.session_state.df = pd.DataFrame(df)
        #     st.session_state.dl.append(st.session_state.df)
        # st.success("Data uploaded successfully!")

# if "langchain_messages" not in st.session_state:
#     st.session_state["langchain_messages"] = []

## display all the dfs
# if len(st.session_state.dl) > 0:
#     for df in st.session_state.dl:
#         st.write(df.head())

## only add the newly uploaded dfs
# if len(st.session_state.dl) > 0:
#     num_new_files = len(st.session_state.uploaded_files) if st.session_state.uploaded_files else 0
#     start_index = len(st.session_state.dl) - num_new_files
#     for df in st.session_state.dl[start_index:]:
#         st.write(df.head())

# for msg in msgs.messages:
#     st.chat_message(msg.type).write(msg.content)

# Render current messages from StreamlitChatMessageHistory
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

prompt = st.chat_input("Your question here: ")

if prompt:
    if len(st.session_state.dl) == 0:
        st.error("Please click Process before using chatbot.")
    else:
        # Note: new messages are saved to history automatically by Langchain during run
        config = {"configurable": {"session_id": "any"}}
        llm = OpenAI(api_token=st.session_state.openai_key, model="gpt-4o")
        agent = Agent(st.session_state.dl,
                      config={"llm": llm,
                              "enforce_privacy": True,
                              "response_parser": StreamlitResponse},
                      memory_size=10)
        response = agent.chat(prompt)
        explain = agent.explain()

        msgs.add_user_message(prompt)
        # msgs.add_ai_message(f"{response} \n\n {explain}")
        st.chat_message("user").write(prompt)
        st.chat_message("ai").write(f"{response} \n\n {explain}")

        fig = plt.gcf()

        if fig.get_axes():
            st.chat_message("ai").write("")
            st.pyplot(fig)
            msgs.add_ai_message(explain)
            # msgs.add_ai_message("The response is an image, can't be saved.")
        elif isinstance(response, pd.DataFrame):
            st.chat_message("ai").write("")
            if len(response) > 20:
                st.table(response.head(5))  # Display head of table
            else:
                st.table(response)  # Display whole tables
            # msgs.add_ai_message("The response is a table,can't be saved.")
            msgs.add_ai_message(explain)
        else:
            # st.chat_message("ai").write(explain)  # Display all others
            msgs.add_ai_message(f"{response} \n\n {explain}")
