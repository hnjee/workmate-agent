import os
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import Optional
from config import config


def get_embeddings():
    """임베딩 모델을 반환합니다."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )


def load_vectorstore() -> Optional[Chroma]:
    """저장된 벡터스토어를 로드합니다."""
    if not os.path.exists(config.VECTORSTORE_DIR):
        print("[경고] 벡터스토어가 없습니다. ingest.py를 먼저 실행하세요.")
        return None

    vectorstore = Chroma(
        persist_directory=config.VECTORSTORE_DIR,
        embedding_function=get_embeddings(),
    )
    print("벡터스토어 로드 완료")
    return vectorstore