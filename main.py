import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from streamlit_extras.buy_me_a_coffee import button

import streamlit as st
import tempfile
import os

button(username="ru6300", floating=True, width=221)

# 제목
st.title("ChatPDF")
st.write("---")

#OpenAI KEY 입력 받기
openai_key = st.text_input('OPEN_AI_API_KEY', type="password")

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
        length_function = len
    )
    texts = text_splitter.split_documents(pages)

    #Embedding
    embeddings_model = OpenAIEmbeddings(openai_api_key=openai_key)

    # Chroma 데이터베이스에 저장
    with tempfile.TemporaryDirectory() as temp_dir:
        db = Chroma.from_documents(texts, embeddings_model, persist_directory=temp_dir)

        # 질문 입력
        st.header("PDF에게 질문해보세요!!")
        question = st.text_input('질문을 입력하세요')

        if st.button('질문하기'):
            with st.spinner('잠시만 기다려 주세요...'):
                llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0,  openai_api_key=openai_key, streaming=True, callbacks=[stream_hander])
                qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())
                result = qa_chain({"query": question})
                st.write(result["result"])
