"""
main.py: WorkMate Agent 실행 진입점

사용법:
    uv run python main.py

실행 전 준비사항:
    1. .env 파일에 API 키 설정
    2. docs/ 디렉토리에 문서 파일 추가
    3. uv run python ingest.py 실행 (문서 임베딩)
"""
from langchain_core.messages import HumanMessage
from src.agent import agent_graph
from config import config
from ingest import main as ingest_main

import os


def validate_config():
    """필수 환경변수가 설정되었는지 확인합니다."""
    if not config.OPENAI_API_KEY:
        print("[에러] OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
        return False
    if not config.TAVILY_API_KEY:
        print("[경고] TAVILY_API_KEY가 설정되지 않았습니다. 웹 검색이 동작하지 않습니다.")
    return True


def run_agent(user_input: str) -> str:
    """에이전트를 실행하고 최종 답변을 반환합니다."""
    initial_state = {
        "messages": [HumanMessage(content=user_input)]
    }

    try:
        final_state = agent_graph.invoke(
            initial_state,
            config={"recursion_limit": config.MAX_ITERATIONS}
        )
        return final_state["messages"][-1].content

    except Exception as e:
        return f"[에러] 에이전트 실행 중 문제가 발생했습니다: {str(e)}"


def main():
    if not os.path.exists(config.VECTORSTORE_DIR):
        print("벡터DB가 없어서 자동으로 문서를 임베딩합니다...")
        ingest_main()

    print("=" * 60)
    print("  WorkMate Agent")
    print("  종료하려면 'quit' 또는 'exit'를 입력하세요.")
    print("=" * 60)

    if not validate_config():
        return

    print("\n준비 완료! 질문을 입력하세요.\n")

    while True:
        try:
            user_input = input("질문: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n에이전트를 종료합니다.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "종료"):
            print("에이전트를 종료합니다.")
            break

        print("\n[답변 생성 중...]\n")
        answer = run_agent(user_input)
        print(f"답변:\n{answer}")
        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()