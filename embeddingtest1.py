import pandas as pd
from langchain_community.vectorstores.faiss import FAISS
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_upstage import UpstageEmbeddings
from bs4 import BeautifulSoup
import re, tiktoken, os

# 토크나이저 설정
tokenizer = tiktoken.get_encoding("cl100k_base")
def tiktoken_len(text) :
    tokens = tokenizer.encode(text)
    return len(tokens)

# Step 1: Excel 파일에서 URL 읽어오기
def load_data_from_excel(file_path):
    df = pd.read_excel(file_path)
    urls = df['url'].tolist()
    metadata = df[['kb_title', 'source', 'url']].to_dict(orient='records')
    return urls, metadata

# Step 2: HTML 태그 제거와 특정 패턴의 텍스트 삭제를 위한 클렌징 함수
def clean_text(text):
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()  # HTML 태그 제거
    text = re.sub(r'\s+', ' ', text)  # 연속된 공백 제거

    # Keywords Backlink 이후의 텍스트 제거
    text = re.split(r'Keywords', text)[0]

    # "발행 : YYYY. MM. DD" 및 "수정 : YYYY. MM. DD" 형식의 텍스트 제거
    text = re.sub(r'발행 : \d{4}\. \d{2}\. \d{2}', '', text)
    text = re.sub(r'수정 : \d{4}\. \d{2}\. \d{2}', '', text)

    # "Toggle navigation Login"과 같은 불필요한 텍스트 제거
    text = re.sub(r'Toggle navigation Login', '', text)

    return text.strip()

# Step 3: 웹페이지 로드 및 텍스트 불러오기, 클렌징 적용
def load_and_clean_web_pages(urls):
    documents = []
    for url in urls:
        loader = WebBaseLoader(url)
        try:
            data = loader.load()
            if data:  # data가 있는 경우만 추가
                cleaned_data = [clean_text(doc.page_content) for doc in data]  # HTML 클렌징 적용
                documents.extend(cleaned_data)
        except Exception as e:
            print(f"Failed to load {url}: {e}")
    return documents

# Step 4: 텍스트를 하나의 chunk로 생성하고 확인
def generate_chunks(documents):
    # Text Splitter 생성 (웹페이지 당 하나의 chunk로 지정)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,  # 설정을 크게 해서 웹페이지 하나가 하나의 chunk가 되도록 함
        chunk_overlap=0,
        length_function=len,
    )

    # 각 문서를 chunk로 나눔
    chunks = [text_splitter.split_text(doc) for doc in documents]
    
    # Flattening chunks list (중첩된 리스트 구조 풀기)
    flat_chunks = [chunk for sublist in chunks for chunk in sublist]
    
    return flat_chunks

# Step 5: 생성된 chunks를 FAISS에 저장
def save_chunks_to_faiss(flat_chunks, metadata):
  embeddings = UpstageEmbeddings(
      api_key="up_1S1HRRiWDnK9FR8ssqJsKFdwsTPbq",
      model="embedding-passage")
#   model_name = "jhgan/ko-sbert-nli"
#   model_kwargs = {'device': 'cpu'}
#   encode_kwargs = {'normalize_embeddings': True}
#   hf = HuggingFaceEmbeddings(
#       model_name=model_name,
#       model_kwargs=model_kwargs,
#       encode_kwargs=encode_kwargs
#   )
    
  # FAISS 벡터 스토어에 chunk 임베딩 저장
  vectorstore = FAISS.from_texts(flat_chunks, embeddings, metadatas=metadata)
  vectorstore.save_local("QA_DB(241121)_ups")

  return vectorstore

# 전체 코드 실행 예시
file_path = "C:\python-RAG\doc_QA\page_list.xlsx"  # 엑셀 파일 경로 설정
urls, metadata = load_data_from_excel(file_path)
documents = load_and_clean_web_pages(urls)
chunks = generate_chunks(documents)

# 메타데이터 수가 chunk 수와 맞지 않을 수 있어 일치하도록 조정
metadata_repeated = metadata * (len(chunks) // len(metadata)) + metadata[:len(chunks) % len(metadata)]
vectorstore = save_chunks_to_faiss(chunks, metadata_repeated)

# 저장된 벡터 스토어 확인
print("FAISS DB 저장 완료.")