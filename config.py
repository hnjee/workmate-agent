import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

    # 문서 경로
    DOCS_DIR: str = os.path.join(os.path.dirname(__file__), "docs")

    # 벡터스토어 경로
    VECTORSTORE_DIR: str = os.path.join(os.path.dirname(__file__), ".vectorstore")

    # RAG 설정
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 100
    TOP_K: int = 3  # 검색 시 반환할 문서 수

    # LLM 설정
    MODEL_NAME: str = "gpt-4o"
    MAX_ITERATIONS: int = 10  # 에이전트 최대 루프 횟수 (무한루프 방지)


config = Config()
