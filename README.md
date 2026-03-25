# 신입 개발자를 위한 사내 기술 가이드 & 이슈 해결 에이전트

## 프로젝트 개요

실무에서 쌓아온 업무 기록(마크다운, PDF)을 기반으로, 개발 이슈를 질문하면 내부 문서와 외부 웹을 스스로 판단해 검색하고 답변을 생성하는 AI 에이전트입니다.

## 기술 스택

| 항목 | 기술 |
|------|------|
| 언어 | Python 3.11+ |
| 에이전트 프레임워크 | LangGraph |
| LLM | Claude 3.5 Sonnet (Anthropic) |
| 내부 문서 검색 | LangChain RAG + ChromaDB |
| 외부 검색 | Tavily Search API |
| 2단계 MCP | Notion, Slack |
| 3단계 MCP | Gmail |

## 프로젝트 구조

```
tech-guide-agent/
│
├── docs/                        # 내부 문서 (업무 기록, PDF)
│   ├── langchain_notes.md
│   ├── langgraph_notes.md
│   └── company_guide.pdf
│
├── src/
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── graph.py             # LangGraph 에이전트 그래프 정의
│   │   ├── state.py             # 에이전트 상태 정의
│   │   └── nodes.py             # 그래프 노드 함수들
│   │
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── internal_search.py   # RAG 기반 내부 문서 검색 툴
│   │   └── web_search.py        # 웹 검색 툴
│   │
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── loader.py            # 문서 로더 (마크다운, PDF)
│   │   └── vectorstore.py       # 벡터스토어 초기화 및 관리
│   │
│   └── config.py                # 환경변수 및 설정
│
├── main.py                      # 실행 진입점
├── ingest.py                    # 문서 임베딩 및 벡터DB 저장
├── requirements.txt
└── .env
```

## 에이전트 흐름도

```
사용자 질문
     │
     ▼
[agent 노드] ── LLM이 도구 선택 판단
     │
     ├── search_internal_docs 선택
     │        │
     │        ▼
     │   [RAG 검색] ── 벡터DB에서 유사 문서 검색
     │        │
     │        └── 결과 반환 → agent로 복귀
     │
     ├── search_web 선택
     │        │
     │        ▼
     │   [Tavily 검색] ── 외부 웹 검색
     │        │
     │        └── 결과 반환 → agent로 복귀
     │
     └── 도구 없이 최종 답변 생성
              │
              ▼
         [END] 사용자에게 응답
```

## Agent다움의 핵심 설계 포인트

### 1. 도구 선택 판단
LLM이 질문의 성격을 보고 스스로 결정합니다.
- "langchain 메모리 에러" → 내부 문서 먼저 검색
- "최신 LangGraph 버전" → 웹 검색
- 내부에 없으면 → 웹 검색으로 자동 전환

### 2. 결과 평가 루프
검색 결과가 질문과 관련 없거나 불충분하면, LLM이 판단하여 다른 도구를 추가로 사용합니다.

### 3. 에러 처리 (Graceful Fallback)
- 도구 실행 실패 시 에러를 Agent에게 알려 다른 방법을 시도하게 합니다.
- 모든 도구가 실패해도 "찾을 수 없음"을 명확히 안내합니다.

### 4. Human-in-the-loop (2단계)
Slack 전송 전 반드시 사용자 확인을 거칩니다.

## 단계별 로드맵

### MVP (현재)
- [x] 로컬 문서 RAG 툴
- [x] 웹 검색 툴
- [x] LangGraph 기반 도구 선택 + 루프
- [x] 에러 처리

### 2단계
- [ ] Notion MCP 연결 (RAG 대체)
- [ ] Slack MCP 연결 + Human-in-the-loop

### 3단계
- [ ] Gmail MCP 연결

## 환경변수 (.env)

```
ANTHROPIC_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here
```
