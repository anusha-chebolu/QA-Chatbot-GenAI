import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

## Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Simple Q&A Chatbot With OPENAI"

## Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user queries"),
        ("user", "Question: {question}")
    ]
)

def generate_response(question, api_key, engine, temperature, max_tokens):
    os.environ["OPENAI_API_KEY"] = api_key  # Set API key in environment for LangChain

    # Initialize the language model
    llm = ChatOpenAI(model=engine, temperature=temperature, max_tokens=max_tokens)
    
    # Create the chain
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    
    # Get the answer
    answer = chain.invoke({'question': question})
    
    # Ensure answer is in string format
    return str(answer)

# Title of the app
st.title("Enhanced Q&A Chatbot With OpenAI")

# Sidebar for settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")

# Select the OpenAI model
engine = st.sidebar.selectbox("Select OpenAI model", ["gpt-4", "gpt-4o", "gpt-3.5-turbo"])

# Adjust response parameters
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

# Main interface for user input
st.write("Go ahead and ask any question")
user_input = st.text_input("You:")

if user_input and api_key:
    # Generate the response and display it
    response = generate_response(user_input, api_key, engine, temperature, max_tokens)
    st.write("Bot:", response)

elif user_input:
    st.warning("Please enter the OpenAI API Key in the sidebar.")
else:
    st.write("Please provide the user input.")
