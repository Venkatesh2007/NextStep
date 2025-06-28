import streamlit as st
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS


import os
from dotenv import load_dotenv
load_dotenv()

HF_TOKEN = st.secrets["HF_TOKEN"]
embeddings = HuggingFaceEmbeddings(model="all-MiniLM-L6-v2")

## Title of the app
st.set_page_config(page_title="NextStep",page_icon="🦜")

# Sidebar for controls
with st.sidebar:
    st.title("Controls")
    uploaded_file = st.file_uploader("Upload your PDF Resume", type="pdf")
    session_id = st.text_input("Session ID", value="default_session")
    groq_api_key = st.secrets["GROQ_API_KEY"]
    if not groq_api_key:
        st.warning("Please set your GROQ_API_KEY in environment variables.")

st.title("NextStep")
st.write("Career Clarity for the Under-Guided")

if not groq_api_key:
    st.error("GROQ API Key not found. Please set it in your environment variables.")
    st.stop()

llm = ChatGroq(groq_api_key=groq_api_key, model="Gemma2-9b-It")

# Initialize chat history storage in session state
if 'store' not in st.session_state:
    st.session_state.store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

documents = []
if uploaded_file:
    temp_pdf_path = "./temp.pdf"
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_pdf_path)
    documents = loader.load()

if documents:
    # Prepare vectorstore
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. DO NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    system_prompt = (
    "You are NextStep(yourname), a highly knowledgeable and empathetic AI career mentor "
    "designed to help users make smart, personalized decisions about their career paths.\n\n"
    
    "You have access to the user's resume and background information, provided below as context. "
    "Use this to tailor your advice specifically to the user's current skills, experience, and interests.\n\n"
    
    "When answering questions, follow these rules:\n"
    "1. Be clear, concise, and actionable.\n"
    "2. Use a friendly and encouraging tone, like a real human mentor.\n"
    "3. If you're giving a career roadmap or skill path, break it down step-by-step.\n"
    "4. Always consider the user's current level, and suggest achievable next steps.\n"
    "5. If the query is about a specific company or job role (e.g., 'data scientist at Google'), "
    "provide relevant skills, experiences, certifications, and projects needed to get there.\n"
    "6. If you don’t have enough information to provide an answer, say so honestly and recommend what the user should clarify or explore.\n\n"
    
    "Context (User's Resume and Profile):\n{context}\n\n"
    "Now, based on the above, answer the user's question in a structured and helpful manner."
)

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain, get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    session_history = get_session_history(session_id)

    # Display all chat messages with st.chat_message
    for msg in session_history.messages:
        if msg.type == "human":
            with st.chat_message("user"):
                st.markdown(msg.content)
        else:
            with st.chat_message("assistant"):
                st.markdown(msg.content)

    # User input box
    user_input = st.chat_input("Your Question:")
    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                response = conversational_rag_chain.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": session_id}}
                )
                answer = response["answer"]
                st.markdown(answer)

else:
    st.info("Please upload a PDF resume file in the sidebar to start chatting.")
