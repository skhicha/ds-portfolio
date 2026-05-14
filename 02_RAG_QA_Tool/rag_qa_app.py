import streamlit as st
import pdfplumber
import io, os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

st.set_page_config(page_title="Document Q&A (RAG)", page_icon="🤖", layout="wide")
st.title("🤖 LLM-Powered Document Q&A")

api_key = st.sidebar.text_input("Google Gemini API Key", type="password")
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

@st.cache_resource
def build_vectorstore(file_bytes_tuple):
    all_text = ""
    for content in file_bytes_tuple:
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            all_text += " ".join(p.extract_text() or "" for p in pdf.pages)
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)  # ← fixed
    chunks = splitter.split_text(all_text)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    return FAISS.from_texts(chunks, embeddings)

PROMPT_TEMPLATE = (
    "You are a precise document assistant. Answer ONLY from the context below.\n"
    "If the answer is not present, say 'Not found in document.'\n\n"
    "Context: {context}\n"
    "Question: {question}\n"
    "Answer:"
)

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

uploaded = st.file_uploader("Upload PDF documents", type="pdf", accept_multiple_files=True)

if uploaded and api_key:
    file_bytes = tuple(f.read() for f in uploaded)
    with st.spinner("Building vector store..."):
        vectorstore = build_vectorstore(file_bytes)
    st.success(f"Indexed {len(uploaded)} document(s). Start asking questions!")

    llm       = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0)
    prompt    = PromptTemplate.from_template(PROMPT_TEMPLATE)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if question := st.chat_input("Ask a question about your documents..."):
        st.chat_message("user").write(question)
        with st.spinner("Searching document..."):
            answer      = rag_chain.invoke(question)
            source_docs = retriever.invoke(question)
        st.chat_message("assistant").write(answer)
        with st.expander("Source Chunks Used"):
            for i, doc in enumerate(source_docs):
                st.markdown(f"**Chunk {i+1}:** {doc.page_content}")
        st.session_state.messages.extend([
            {"role": "user",      "content": question},
            {"role": "assistant", "content": answer},
        ])
elif uploaded and not api_key:
    st.warning("Enter your Google Gemini API key in the sidebar to start querying.")
else:
    st.info("Upload one or more PDF documents to get started.")
