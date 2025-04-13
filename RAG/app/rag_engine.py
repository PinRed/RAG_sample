from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
import os

class RAGEngine:
    def __init__(self):
        self.qa_chain = self._build_chain()

    def _build_chain(self):
        # 載入與切分文本
        loader = TextLoader("docs/knowledge.txt", encoding="utf-8")
        documents = loader.load()
        splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        texts = splitter.split_documents(documents)

        # Embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # FAISS 向量資料庫
        db = FAISS.from_documents(texts, embeddings)
        retriever = db.as_retriever()

        # 載入本地 LLM 模型
        llm = LlamaCpp(
            model_path="models/llama-2-7b-chat.Q4_K_M.gguf",
            temperature=0.5,
            max_tokens=512,
            top_p=0.95,
            n_ctx=2048,
            verbose=True,
        )

        # 建立 RAG QA 鏈
        return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    def ask(self, query: str) -> str:
        return self.qa_chain.run(query)
