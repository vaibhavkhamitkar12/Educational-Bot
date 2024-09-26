import streamlit as st
from workflow import app
from langchain_core.messages import HumanMessage, AIMessage
from IPython.core.display import Markdown

# Function to interact with the workflow
def run_workflow(user_query: str, chat_history: list):
    app_state = {
        "query": user_query,
        "chat_history": chat_history,
        "generation": None,
        "documents": None
    }
    response = app.invoke(app_state)
    return response

# Streamlit UI
st.title("RAG-Based Retrieval Chatbot")

# Create a session state for storing chat history
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Input box for user query
user_query = st.text_input("Enter your query:")

# Button to run the workflow
if st.button("Submit Query"):
    if user_query:
        # Get response from workflow
        response = run_workflow(user_query, st.session_state['chat_history'])

        # Display the response
        st.markdown(Markdown(response['generation']))

        # Add the user query and response to the chat history
        st.session_state['chat_history'].append(HumanMessage(user_query))
        st.session_state['chat_history'].append(AIMessage(response['generation']))
    else:
        st.error("Please enter a query.")

# Display chat history
if st.session_state['chat_history']:
    st.subheader("Chat History:")
    for msg in st.session_state['chat_history']:
        if isinstance(msg, HumanMessage):
            st.write(f"**You:** {msg.content}")
        else:
            st.write(f"**Bot:** {msg.content}")
