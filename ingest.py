"""
ingest.py: docs/ 디렉토리의 문서를 임베딩하여 벡터DB에 저장합니다.
에이전트 실행 전에 반드시 먼저 실행해야 합니다.

사용법:
    uv run python ingest.py
"""
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from typing import List
from config import config
from vectorstore import get_embeddings


# ── 1. 문서 로드 ──────────────────────────────────────────────

def load_documents() -> List[Document]:
    """docs/ 디렉토리의 마크다운, PDF 파일을 모두 로드합니다."""
    documents = []

    if not os.path.exists(config.DOCS_DIR):
        print(f"[경고] docs/ 디렉토리가 없습니다: {config.DOCS_DIR}")
        return []

    for filename in os.listdir(config.DOCS_DIR):
        filepath = os.path.join(config.DOCS_DIR, filename)

        try:
            if filename.endswith(".md") or filename.endswith(".txt"):
                loader = TextLoader(filepath, encoding="utf-8")
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source_file"] = filename
                documents.extend(docs)
                print(f"[로드 완료] {filename}")

            elif filename.endswith(".pdf"):
                loader = PyPDFLoader(filepath)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source_file"] = filename
                documents.extend(docs)
                print(f"[로드 완료] {filename} ({len(docs)}페이지)")

        except Exception as e:
            print(f"[에러] {filename} 로드 실패: {e}")
            continue

    print(f"\n총 {len(documents)}개 문서 로드 완료")
    return documents


# ── 2. 청크 분할 ──────────────────────────────────────────────

def split_documents(documents: List[Document]) -> List[Document]:
    """문서를 청크로 분할합니다."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"총 {len(chunks)}개 청크로 분할 완료")
    return chunks

# ── 3. 벡터DB 생성 ────────────────────────────────────────────

def build_vectorstore(chunks: List[Document]) -> Chroma:
    """청크를 임베딩하여 벡터스토어를 생성하고 저장합니다."""
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=config.VECTORSTORE_DIR,
    )
    print(f"벡터스토어 저장 완료: {config.VECTORSTORE_DIR}")
    return vectorstore


# ── 실행 ──────────────────────────────────────────────────────

def main():
    print("=" * 50)
    print("문서 임베딩 시작")
    print("=" * 50)

    print(f"\n[1/3] 문서 로드 중... ({config.DOCS_DIR})")
    documents = load_documents()

    if not documents:
        print("\n[중단] 로드된 문서가 없습니다.")
        print("docs/ 디렉토리에 .md, .txt, .pdf 파일을 추가한 후 다시 실행하세요.")
        return

    print(f"\n[2/3] 청크 분할 중...")
    chunks = split_documents(documents)

    print(f"\n[3/3] 임베딩 생성 및 벡터스토어 저장 중...")
    build_vectorstore(chunks)

    print("\n" + "=" * 50)
    print("완료! 이제 uv run python main.py로 에이전트를 실행하세요.")
    print("=" * 50)


if __name__ == "__main__":
    main()