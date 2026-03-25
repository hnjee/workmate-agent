from typing import Annotated, List
from langchain.schema import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """
    LangGraph 에이전트의 상태를 정의합니다.

    messages: 대화 기록 전체 (add_messages로 자동 누적)
    """
    # add_messages: 새 메시지를 기존 리스트에 자동으로 추가해주는 reducer
    messages: Annotated[List[BaseMessage], add_messages]
