from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from config import config
from .utils import TOOLS
from .state import AgentState
from langgraph.prebuilt import tools_condition


# ── LLM 초기화 ────────────────────────────────────────────────

llm = ChatOpenAI(
    model=config.MODEL_NAME
).bind_tools(TOOLS)

tool_node = ToolNode(TOOLS)

SYSTEM_PROMPT = """당신은 개발자를 위한 기술 가이드 에이전트입니다.

## 역할
개발 관련 질문에 대해 내부 문서와 외부 웹을 검색하여 정확하고 실용적인 답변을 제공합니다.

## 도구 사용 원칙
1. **내부 문서 우선**: 직접 경험하거나 정리한 내용과 관련 있으면 `search_internal_docs`를 먼저 사용하세요.
2. **웹 검색 보완**: 내부 문서에서 충분한 답을 찾지 못했거나 최신 정보가 필요하면 `search_web`을 사용하세요.
3. **결과 평가**: 검색 결과가 질문과 관련 없거나 불충분하다면 다른 검색어로 다시 시도하세요.
4. **솔직한 한계 인정**: 모든 도구를 사용해도 답을 찾지 못하면 찾지 못했다고 명확히 알려주세요.

## 답변 형식
- 출처(내부문서/웹)를 명시하세요.
- 코드 예시가 있으면 반드시 포함하세요.
- 불확실한 내용은 "확인이 필요합니다"라고 표시하세요.
"""


# ── Nodes ─────────────────────────────────────────────────────

def agent_node(state: AgentState) -> AgentState:
    """LLM이 상태를 보고 다음 행동(도구 사용 or 최종 답변)을 결정합니다."""
    messages = state["messages"]

    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    response = llm.invoke(messages)
    return {"messages": [response]}


# ── Graph ─────────────────────────────────────────────────────

def build_graph():
    """
    LangGraph 에이전트 그래프를 구성하고 컴파일합니다.

    흐름:
        START → agent → (조건 판단)
                            ├── tools → agent (루프)
                            └── end
    """
    graph = StateGraph(AgentState)


    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        tools_condition
    )
    graph.add_edge("tools", "agent")

    return graph.compile()


agent_graph = build_graph()