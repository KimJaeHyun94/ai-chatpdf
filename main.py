import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chroma import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
import streamlit as st
import tempfile
import os

# 제목
st.title("ChatPDF")
st.write("---")

# 파일 업로드
uploaded_file = st.file_uploader("PDF 파일을 올려주세요!", type=['pdf'])
st.write("---")

def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages

# 업로드 되면 동작하는 코드
if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)

    # 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 300,
        chunk_overlap = 20,
        length_function = len,
        is_separator_regex = False,
    )
    texts = text_splitter.split_documents(pages)

    # 임베딩 생성
    embeddings_model = OpenAIEmbeddings()

    # Chroma 데이터베이스에 저장
    with tempfile.TemporaryDirectory() as temp_dir:
        db = Chroma.from_documents(texts, embeddings_model, persist_directory=temp_dir)

        # 질문 입력
        st.header("PDF에게 질문해보세요!!")
        question = st.text_input('질문을 입력하세요')

        if st.button('질문하기'):
            with st.spinner('잠시만 기다려 주세요...'):
                llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)
                qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())
                result = qa_chain({"query": question})
                st.write(result["result"])
