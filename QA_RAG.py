from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_upstage import UpstageEmbeddings

from langchain_community.document_transformers import LongContextReorder
from langchain.retrievers.multi_query import MultiQueryRetriever

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

import os

from dotenv import load_dotenv
load_dotenv()

# langsmith 연동
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"]="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"
os.environ["LANGCHAIN_PROJECT"]="지식카드 작업"

################### for streaming in Streamlit without LECL ###################
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)
# stream_handler = StreamHandler(st.empty())
""" if you want to use streaming on your streamlit app, it's tricky to seperate model script \n
and streamlit script if not using LECL, because llm will have to use 'streaming=True' \n
and 'callbacks=[stream_handler]' and streamhandler uses st.empty() placeholder here which can't be first streamlit command. 
"""
####################### EMBEDDINGS #################################

# embedding = HuggingFaceEmbeddings(
#     model_name="jhgan/ko-sbert-nli",
#     model_kwargs={'device': 'cpu'},
#     encode_kwargs={'normalize_embeddings': True}
# )

embeddings = UpstageEmbeddings(api_key="up_1S1HRRiWDnK9FR8ssqJsKFdwsTPbq", model="embedding-passage")

db = FAISS.load_local(
    folder_path="DB(241216)_test",
    embeddings=embeddings,
    allow_dangerous_deserialization=True
)

########## LLM & Retriever ##########

llm = ChatOpenAI(model_name="gpt-4o", temperature=0.2, streaming=True)

long_context_reorder = LongContextReorder()

retriever = db.as_retriever(search_type="similarity", search_kwargs={'k': 5, 'fetch_k': 10}, document_transformer = long_context_reorder)

multiquery_retriever = MultiQueryRetriever.from_llm(retriever = retriever, llm = llm)

########## Rerank 설정 ##########

# CrossEncoder 모델 초기화
reranker_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")

# 상위 5개의 문서를 선택하도록 CrossEncoderReranker 설정
compressor = CrossEncoderReranker(model=reranker_model, top_n=3)

# 기존의 multiquery_retriever를 ContextualCompressionRetriever로 래핑
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, 
    base_retriever=multiquery_retriever
)

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

############################################## RAG ########################################################

########## Creating prompt ##########
prompt_template = """You are an expert in fire safety, evacuation, and fire prevention. The person asking questions is also a fire safety, evacuation, and fire prevention professional with basic knowledge.
                     Respond based on the relevant regulations provided by the user. You may use line breaks or numbering in your explanation to make the content easier to read.
                     Include detailed information in your response without summarizing, ensuring that no essential content is omitted. If you don't know the answer, just say that you don't know, don't try to make up an answer. Answer in Korean.
                     Do not add unnecessary comments; focus on technical content.

{context}

Question: {question}
Helpful Answer:
"""

prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

########## RAG's chain in langchain's LECL format ##########
rag_chain = ({"context": compression_retriever | format_docs, "question": RunnablePassthrough()} | 
             prompt | llm | StrOutputParser())

def inference(query: str):
    return rag_chain.stream(query)

def retrieve_documents(query: str):
    documents = multiquery_retriever.get_relevant_documents(query)  # documents 검색
    return documents  # 검색된 문서 반환

import os

print("OpenAI API Key:", os.getenv("OPENAI_API_KEY"))
