from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_transformers import LongContextReorder
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from dotenv import load_dotenv
import os

load_dotenv()

# # langsmith 연동
# os.environ["LANGCHAIN_TRACING_V2"]="true"
# os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
# os.environ["LANGCHAIN_API_KEY"]="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"
# os.environ["LANGCHAIN_PROJECT"]="Tech_QA"

################### for streaming in Streamlit without LECL ###################
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)
        
# --------------------------------------------------------------------------
#    전역 변수들: 최초에는 None
# --------------------------------------------------------------------------
embedding = None
db = None
llm = None
retriever = None
multiquery_retriever = None
compression_retriever = None
rag_chain = None

# --------------------------------------------------------------------------
#    초기화 함수
# --------------------------------------------------------------------------
def initialize_rag():
    """
    Lazy-loading 방식으로, 필요할 때 한 번만 모델/DB/체인 등을 로딩하고 초기화한다.
    """
    global embedding, db, llm, retriever
    global multiquery_retriever, compression_retriever, rag_chain

    if embedding is None:
        embedding = HuggingFaceEmbeddings(
            model_name="jhgan/ko-sbert-nli",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

    if db is None:
        db = FAISS.load_local(
            folder_path="C:\\python_rag\\database\\DB(241216)_fire_water",
            embeddings=embedding,
            allow_dangerous_deserialization=True
        )

    if llm is None:
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2, streaming=True)

    if retriever is None:
        long_context_reorder = LongContextReorder()
        base_retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={'k': 5, 'fetch_k': 10},
            document_transformer=long_context_reorder
        )
        mqr = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=llm
        )
        multiquery_retriever = mqr

    if compression_retriever is None:
        # CrossEncoder 모델 초기화
        reranker_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
        compressor = CrossEncoderReranker(model=reranker_model, top_n=5)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, 
            base_retriever=multiquery_retriever
        )

    if rag_chain is None:
        # RAG Chain 구성
        prompt_template = """
        You are an expert in fire safety, evacuation, and fire prevention. 
        The person asking questions is also a fire safety, evacuation, and fire prevention professional with basic knowledge.
        Respond based on the relevant regulations provided by the user. 
        You may use line breaks or numbering in your explanation to make the content easier to read. 
        Include detailed information in your response without summarizing, ensuring that no essential content is omitted. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer. 
        Answer in Korean. your name is Bridge 기술챗봇 for 수계소화설비.

        {context}

        Question: {question}
        Helpful Answer:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

        rag_chain_local = (
            {"context": multiquery_retriever | format_docs, "question": RunnablePassthrough()} 
            | prompt 
            | llm 
            | StrOutputParser()
        )
        rag_chain = rag_chain_local

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

def inference(query: str):
    if rag_chain is None:
        initialize_rag()
    return rag_chain.stream(query)


def retrieve_documents(query: str):
    if multiquery_retriever is None:
        initialize_rag()
    documents = multiquery_retriever.get_relevant_documents(query)
    return documents

# --------------------------------------------------------------------------
#    실제 메인 스크립트로 실행될 때만 돌릴 코드
# --------------------------------------------------------------------------
if __name__ == '__main__':
    # 여기서 초기화 함수를 호출하거나, 테스트용으로 inference를 직접 호출할 수 있다.
    initialize_rag()
    print("RAG System Initialized. You can now call inference(query).")
