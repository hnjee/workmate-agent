from langchain.tools import tool
from tavily import TavilyClient
from config import config
from vectorstore import load_vectorstore

# ── 벡터스토어 싱글턴 ─────────────────────────────────────────
# 매 요청마다 로드하지 않도록 전역으로 한 번만 로드
_vectorstore = None

def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = load_vectorstore()
    return _vectorstore


# ── 내부 문서 검색 툴 (RAG) ───────────────────────────────────

@tool #커스텀 로직 필요
def search_internal_docs(query: str) -> str:
    """
    내부 업무 문서(마크다운, PDF)에서 관련 내용을 검색합니다.
    직접 정리한 기술 노트나 회사 가이드 문서를 찾을 때 사용하세요.
    최신 정보보다는 내가 직접 경험하고 정리한 내용을 찾을 때 적합합니다.
    """
    vectorstore = get_vectorstore()

    if vectorstore is None:
        return "[에러] 벡터스토어를 불러올 수 없습니다. ingest.py를 먼저 실행해주세요."

    try:
        results = vectorstore.similarity_search(query, k=config.TOP_K)

        if not results:
            return "내부 문서에서 관련 내용을 찾지 못했습니다."

        formatted = []
        for i, doc in enumerate(results, 1):
            source = doc.metadata.get("source_file", "알 수 없음")
            page = doc.metadata.get("page", "")
            page_info = f" (p.{page})" if page != "" else ""
            formatted.append(
                f"[내부문서 {i}] 출처: {source}{page_info}\n{doc.page_content}"
            )

        return "\n\n---\n\n".join(formatted)

    except Exception as e:
        return f"[에러] 내부 문서 검색 중 오류 발생: {str(e)}"


# ── 웹 검색 툴 (Tavily) ───────────────────────────────────────

_tavily_client = None

def get_tavily_client():
    global _tavily_client
    if _tavily_client is None:
        if not config.TAVILY_API_KEY:
            raise ValueError("TAVILY_API_KEY가 설정되지 않았습니다.")
        _tavily_client = TavilyClient(api_key=config.TAVILY_API_KEY)
    return _tavily_client


@tool #결과 포맷팅 제어를 위해 
def search_web(query: str) -> str:
    """
    인터넷에서 최신 정보를 검색합니다.
    공식 문서, 최신 버전 정보, 스택오버플로우 답변 등 외부 정보가 필요할 때 사용하세요.
    내부 문서에 없는 내용이거나 최신 정보가 필요할 때 적합합니다.
    """
    try:
        client = get_tavily_client()
        response = client.search(
            query=query,
            max_results=3,
            search_depth="basic",
            include_answer=True,
        )

        results = []

        if response.get("answer"):
            results.append(f"[요약] {response['answer']}")

        for i, item in enumerate(response.get("results", []), 1):
            title = item.get("title", "제목 없음")
            url = item.get("url", "")
            content = item.get("content", "")
            results.append(f"[웹검색 {i}] {title}\n출처: {url}\n{content}")

        if not results:
            return "웹 검색 결과를 찾지 못했습니다."

        return "\n\n---\n\n".join(results)

    except ValueError as e:
        return f"[설정 에러] {str(e)}"
    except Exception as e:
        return f"[에러] 웹 검색 중 오류 발생: {str(e)}"


# 에이전트에 전달할 툴 목록
TOOLS = [search_internal_docs, search_web]