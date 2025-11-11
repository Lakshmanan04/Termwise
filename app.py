import os
import fitz 
import streamlit as st
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate

GEMINI_API_KEY = st.secrets["API_KEY"]

st.set_page_config(page_title="Termwise", layout="wide")
st.title("🧾 Termwise: Understand What You're Agreeing To")

llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", google_api_key=GEMINI_API_KEY, temperature=0.7)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)

def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    return "".join([page.get_text() for page in doc])

def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

def create_vectorstore(chunks):
    docs = [Document(page_content=chunk) for chunk in chunks]
    return FAISS.from_documents(docs, embeddings)

def clear_conversation():
    st.session_state.policy_conversation = []
    st.session_state.qa_conversation = []
    st.session_state.policy_vectorstore = None
    st.session_state.qa_vectorstore = None

def get_conversation_context(conversation_history, max_pairs=5):
    if len(conversation_history) <= max_pairs * 2:
        return conversation_history
    return conversation_history[-(max_pairs * 2):]

def display_conversation(conversation_history):
    for i, message in enumerate(conversation_history):
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])

# ---------- Chains ----------
def summarize_terms(text):
    prompt = PromptTemplate.from_template("""
    Act as a legal assistant. Provide a concise summary of the following terms in 5-7 bullet points.
    Focus on obligations, key terms, and actions users must take or avoid.

    TERMS:
    {text}
    """)
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(text=text[:3000])

def detect_risks(text):
    prompt = PromptTemplate.from_template("""
    You are a risk analysis expert. Review the following document and highlight any risky, unfair, or ambiguous clauses.
    Specifically mention clauses about data sharing, arbitration, cancellation policies, or hidden obligations.

    DOCUMENT:
    {text}
    """)
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(text=text[:3000])

def policy_question_answering(policy_text, question, conversation_context=None):
    context_prompt = ""
    if conversation_context:
        context_str = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in conversation_context])
        context_prompt = f"\n\nPREVIOUS CONVERSATION:\n{context_str}\n"
    
    prompt_template = f"""
    You are an expert assistant trained to understand and interpret organizational and university policies.

    Your task is to answer the user's question accurately based on the following policy text. 
    Only provide information that is clearly supported by the text. If the answer is not explicitly stated or cannot be inferred from the policy, say you don't have enough information.

    Be clear, concise, and formal in your response.{context_prompt}

    POLICY TEXT:
    {{context}}

    USER QUESTION:
    {{question}}

    ANSWER:
    """
    
    prompt = PromptTemplate.from_template(prompt_template)

    chunks = split_text(policy_text)
    vectorstore = create_vectorstore(chunks)
    retriever = vectorstore.as_retriever()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain.run({"query": question})

def document_qa(vectorstore, query, conversation_context=None):
    context_prompt = ""
    if conversation_context:
        context_str = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in conversation_context])
        context_prompt = f"\n\nPREVIOUS CONVERSATION:\n{context_str}\n"
    
    prompt_template = f"""
    You are a helpful assistant that answers questions based on the provided document context.
    Use the context to answer the question accurately. If you cannot find the answer in the context, say so.{context_prompt}

    Context: {{context}}
    Question: {{question}}
    Answer:
    """
    
    prompt = PromptTemplate.from_template(prompt_template)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain({"query": query})

if "selected_tool" not in st.session_state:
    st.session_state.selected_tool = "📄 Terms Analyzer"
if "policy_conversation" not in st.session_state:
    st.session_state.policy_conversation = []
if "qa_conversation" not in st.session_state:
    st.session_state.qa_conversation = []
if "policy_vectorstore" not in st.session_state:
    st.session_state.policy_vectorstore = None
if "qa_vectorstore" not in st.session_state:
    st.session_state.qa_vectorstore = None
if "policy_text_hash" not in st.session_state:
    st.session_state.policy_text_hash = None
if "qa_text_hash" not in st.session_state:
    st.session_state.qa_text_hash = None

st.sidebar.markdown("### 🛠️ Choose a Tool")

tool_options = {
    "📄 Terms Analyzer": "Analyze terms of service or legal text",
    "🏛️ Policy Assistant": "Ask questions about large policies",
    "🔎 Document Q&A": "Query any uploaded document"
}

for name, desc in tool_options.items():
    is_selected = name == st.session_state.selected_tool
    button_style = f"""
        background-color: {'#1f77b4' if is_selected else '#f0f2f6'}; 
        color: {'white' if is_selected else 'black'}; 
        margin-bottom: 0.5rem; 
        width: 100%; 
        border-radius: 0.5rem; 
        text-align: left; 
        padding: 0.5rem 1rem; 
        border: none;
    """
    if st.sidebar.button(f"{name}\n{desc}", key=name, use_container_width=True):
        st.session_state.selected_tool = name

tool = st.session_state.selected_tool

if tool == "📄 Terms Analyzer":
    st.subheader("📄 Terms Analyzer")
    tab1, tab2 = st.tabs(["Upload PDF", "Paste Text"])
    raw_text = ""

    with tab1:
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], key="terms_pdf")
        if uploaded_file:
            with st.spinner("📄 Uploading and extracting text from PDF..."):
                raw_text = extract_text_from_pdf(uploaded_file)

    with tab2:
        raw_text_input = st.text_area("Paste the terms of service or legal content here", height=300)
        if raw_text_input:
            raw_text = raw_text_input

    if raw_text:
        with st.spinner("🔍 Analyzing terms and detecting risks..."):
            summary = summarize_terms(raw_text)
            risks = detect_risks(raw_text)
        st.markdown("### ✅ Summary")
        st.markdown(summary)
        st.markdown("### ⚠️ Risks")
        st.markdown(risks)

elif tool == "🏛️ Policy Assistant":
    st.subheader("🏛️ Policy Assistant")
    
    if st.button("🗑️ Clear Conversation", key="clear_policy"):
        st.session_state.policy_conversation = []
        st.session_state.policy_vectorstore = None
        st.session_state.policy_text_hash = None
        st.rerun()
    
    tab1, tab2 = st.tabs(["Upload PDF", "Paste Text"])
    policy_text = ""
    current_text_hash = None

    with tab1:
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], key="policy_pdf")
        if uploaded_file:
            with st.spinner("📄 Uploading and extracting text from PDF..."):
                policy_text = extract_text_from_pdf(uploaded_file)
                current_text_hash = hash(policy_text)

    with tab2:
        policy_text_input = st.text_area("Paste policy document here", height=300)
        if policy_text_input:
            policy_text = policy_text_input
            current_text_hash = hash(policy_text)

    if current_text_hash != st.session_state.policy_text_hash and current_text_hash is not None:
        st.session_state.policy_conversation = []
        st.session_state.policy_vectorstore = None
        st.session_state.policy_text_hash = current_text_hash

    st.markdown("### 💬 Ask Questions")
    
    if st.session_state.policy_conversation:
        display_conversation(st.session_state.policy_conversation)

    question = st.chat_input(
        "Ask a question about this policy...",
        disabled=not policy_text,
        key="policy_chat_input"
    )

    if question and policy_text:
        st.session_state.policy_conversation.append({"role": "user", "content": question})
        
        with st.chat_message("user"):
            st.write(question)
        
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                context = get_conversation_context(st.session_state.policy_conversation[:-1])
                answer = policy_question_answering(policy_text, question, context)
            
            st.write(answer)
        
        st.session_state.policy_conversation.append({"role": "assistant", "content": answer})

elif tool == "🔎 Document Q&A":
    st.subheader("🔎 Document Q&A")
    
    if st.button("🗑️ Clear Conversation", key="clear_qa"):
        st.session_state.qa_conversation = []
        st.session_state.qa_vectorstore = None
        st.session_state.qa_text_hash = None
        st.rerun()
    
    tab1, tab2 = st.tabs(["Upload PDF", "Paste Text"])
    raw_text = ""
    current_text_hash = None

    with tab1:
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], key="qa_pdf")
        if uploaded_file:
            with st.spinner("📄 Uploading and extracting text from PDF..."):
                raw_text = extract_text_from_pdf(uploaded_file)
                current_text_hash = hash(raw_text)

    with tab2:
        raw_text_input = st.text_area("Paste document text here", height=300)
        if raw_text_input:
            raw_text = raw_text_input
            current_text_hash = hash(raw_text)

    if current_text_hash != st.session_state.qa_text_hash and current_text_hash is not None:
        st.session_state.qa_conversation = []
        st.session_state.qa_vectorstore = None
        st.session_state.qa_text_hash = current_text_hash

    if raw_text and st.session_state.qa_vectorstore is None:
        with st.spinner("🔧 Preparing document for search..."):
            chunks = split_text(raw_text)
            st.session_state.qa_vectorstore = create_vectorstore(chunks)

    st.markdown("### 💬 Ask Questions")
    
    if st.session_state.qa_conversation:
        display_conversation(st.session_state.qa_conversation)

    query = st.chat_input(
        "Ask a question about the document...",
        disabled=not raw_text,
        key="qa_chat_input"
    )

    if query and raw_text and st.session_state.qa_vectorstore:
        st.session_state.qa_conversation.append({"role": "user", "content": query})
        
        with st.chat_message("user"):
            st.write(query)
        
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching..."):
                context = get_conversation_context(st.session_state.qa_conversation[:-1])
                result = document_qa(st.session_state.qa_vectorstore, query, context)
            
            st.write(result["result"])
        
        st.session_state.qa_conversation.append({"role": "assistant", "content": result["result"]})
