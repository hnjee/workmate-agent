# (1)  LangGraph의 핵심 구성요소와 기본 RAG 패턴

강의 주제: LangGraph

## 1. LangGraph란?

- LangGraph는 LLM 기반 워크플로우를 **그래프 구조**로 설계하는 프레임워크다.
    
    
    |  | LangChain | LangGraph |
    | --- | --- | --- |
    | 구조 | 체인 (선형, 단방향) | 그래프 (순환, 분기 가능) |
    | 흐름 제어 | 고정된 순서 | 조건부 분기, 루프 가능 |
    | 적합한 용도 | 단순 파이프라인 | 복잡한 Agent, 반복 로직 |
- LangChain 기반의 orchestration framework
    - LangGraph is a **stateful, orchestration** framework that brings added control to agent workflows
    - orchestration framework: 여러 AI 컴포넌트들(LLM 호출, 도구 사용, 데이터 검색 등)을 정해진 순서와 조건에 따라 조율하고 연결해서 하나의 완성된 AI Agent 워크플로우를 만드는 프레임워크

---

## 2. LangGraph의 3가지 핵심 구성요소

| 구성요소 | 한 줄 역할 |
| --- | --- |
| **(1) State** | 노드 간 데이터를 주고받는 공유 메모리 |
| **(2) Node** | State를 받아 작업하고 변경된 필드를 반환하는 함수 |
| **(3) Edge** | 노드 간 실행 순서를 결정하는 연결선 |
| **Conditional Edge** | 조건에 따라 다음 노드를 동적으로 결정하는 라우터 |

### 1) State — 그래프의 공유 데이터 컨테이너

```python
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]  # 누적(append)
    query: str # 덮어쓰기(update)
    answer: str # 덮어쓰기
```

- 그래프 내 모든 Node가 공유하는 데이터 컨테이너
- 반드시 `dict-like` (키-값 쌍)구조를 상속 (`TypedDict`, `Pydantic`, `dataclass` 등)
- 필드(속성)마다 **업데이트 방식**을 지정할 수 있음

| 업데이트 방식 | 동작 | 사용 예 |
| --- | --- | --- |
| 기본 (덮어쓰기) | 새 값으로 교체 | `query`, `answer` 같은 단일 값 |
| `add_messages` | 기존 리스트에 추가 | `messages` (대화 히스토리) |

### 2) Node — 실제 작업을 수행하는 함수

```python
def my_node(state: AgentState) -> AgentState: # 반드시 state 파라미터 필요
    # ... 작업 수행 
    return {"messages": [new_message]}  # 변경할 필드만 반환, 반드시 dict 타입 
    # LangGraph가 자동으로 기존 state와 리턴 값을 병합, 결과적으로는 AgentState 반환됨
```

- **State를 입력받아 → 변경된 필드만 dict 형태로 반환 →** LangGraph가 반환값을 기존 State와 자동으로 병합
- State 객체를 직접 수정하지 말 것 (반환값으로만 업데이트)
- `return {}` 또는 `return None`이면 State 변경 없음

### 3) Edge — Node 간의 연결과 흐름 제어

| 종류 | 메서드 | 동작 |
| --- | --- | --- |
| 일반 Edge | `add_edge(A, B)` | A 실행 후 항상 B로 이동 |
| 조건부 Edge | `add_conditional_edges(A, fn)` | fn의 반환값에 따라 다음 Node 결정 |
- `START`: 그래프의 시작점 (진입 노드 지정)
- `END`: 그래프의 종료점

---

## 3. 그래프 만드는 순서

```
1. State 정의          → 공유 데이터 구조 설계
2. Node 정의           → 각 작업 함수 구현
3. StateGraph 생성     → 그래프 빌더 초기화
4. Node 등록           → graph_builder.add_node()
5. Edge 연결           → graph_builder.add_edge()
6. 그래프 컴파일         → graph_builder.compile()
7. 실행                → graph.invoke(initial_state)
```

```python
# 한눈에 보는 전체 패턴
graph_builder = StateGraph(AgentState)

graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("generate", generate)

graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("retrieve", "generate")
graph_builder.add_edge("generate", END)

graph = graph_builder.compile()
graph.invoke({"query": "질문"})
```

---

## 4. LangGraph를 활용한 RAG 구축 예시

```python
Step 1: PDF 문서 전처리 
→ Step 2: VectorDB 구축 (Chroma)
→ Step 3: LangGraph로 LLM 답변 가져오기 
```

### 4-1. PDF 전처리 파이프라인

- 사내 문서(PDF)를 그대로 VectorDB에 넣으면 문제가 생김
    - pdf 줄바꿈 시 단어 분리 → 한글 단어 깨짐
    - 이미지로 된 표를 텍스트로 변환 불가 → 표 데이터 누락
- 권장 전처리 흐름

```
PDF → Markdown (zerox, vision 모델 활용) → TXT
```

| 단계 | 도구 | 역할 |
| --- | --- | --- |
| PDF → MD | `py-zerox` (Vision LLM) | 표, 이미지까지 마크다운으로 변환 |
| MD 로드 | `UnstructuredMarkdownLoader` | 마크다운 구조 유지하며 로드 |
| MD → TXT | `markdown` + `html2text` | 마크다운 테이블 활용을 위해 변환 |

### 4-2. VectorDB 구축 (Chroma)

```
전처리된 txt 문서 청크 → Embedding → Chroma VectorDB → Retriever
```

- 청킹: `RecursiveCharacterTextSplitter`
    - chunk_size=1500, overlap=100
- Embedding 모델: `text-embedding-3-large`
- Chroma VectorDB: `persist_directory`로 로컬에 영구 저장 가능
- `retriever = vector_store.as_retriever(search_kwargs={'k': 3})`
    - LangChain의 **Runnable 인터페이스**를 구현한 `VectorStoreRetriever` 객체 리턴
    - LangChain의 표준 방식대로 `.invoke()`, `.batch()`, `.stream()` 등을 바로 사용할 수 있고, LCEL 체인(`|` 연산자)에도 그대로 연결할 수 있음

### 4-3. LangGraph로 구현한 RAG 그래프

- 그래프 설계
    
    ![image.png]((1)%20LangGraph%EC%9D%98%20%ED%95%B5%EC%8B%AC%20%EA%B5%AC%EC%84%B1%EC%9A%94%EC%86%8C%EC%99%80%20%EA%B8%B0%EB%B3%B8%20RAG%20%ED%8C%A8%ED%84%B4/image.png)
    

- State 설계

```python
class AgentState(TypedDict):
    # 사용자 질문 (입력)
    query: str
    # 검색된 문서 (retrieve → generate 전달)
    context: List[Document]
    # 최종 답변 (출력)  
    answer: str           
```

- Node 개발
    - [retrieve] Node: VectorDB에서 관련 문서 검색 → context 업데이트
    - [generate] Node: context + query로 LLM 답변 생성 → answer 업데이트
- Edge 연결: 순차적인 그래프는 `add_edge()` 로 연결

---

## 5. Conditional Edge — 조건부 흐름 제어

### 5-1. Conditional Edge

- `conditional_edge`는 LangGraph에서 조건부 실행 흐름을 제어하는 특별한 종류의 엣지
- 일반 edge와 달리, 특정 조건이 충족될 때만 해당 경로로 실행이 진행됨
- 기본 RAG는 항상 같은 흐름으로 동작하지만, 검색된 문서가 관련 없을 때는 쿼리를 재작성(rewrite) 하는 등 추가적인 액션이 필요하다. 이때 `conditional_edge`를 사용한다.

### 5-2. `add_edge` vs `add_conditional_edge`

- add_edge("출발노드_이름": "도착노드_이름")
    - Node → Node 직접 연결
    - 항상 같은 경로
    - 예: graph_builder.add_edge("rewrite", "retrieve")
- add_conditional_edges("출발노드_이름", 조건_판단_함수)
    - 출발 노드: 실제 작업을 수행하는 노드
    - 조건 판단 함수(조건부 엣지 함수): 다음 노드를 결정하는 라우터
    - 도착 노드들: 함수가 반환하는 문자열에 해당하는 노드들 (자동으로 매핑됨)
    - 예: graph_builder.add_conditional_edges("retrieve", check_doc_relevance)

### 5-3. Node vs 조건부 엣지 함수 비교

|  | Node | 조건부 엣지 함수 |
| --- | --- | --- |
| 역할 | 실제 작업 수행 | 다음 Node 결정 (라우팅만) |
| 반환값 | 업데이트할 State (dict) | 다음 노드 이름 (문자열) |
| 등록 방법 | `add_node()` | `add_conditional_edges()`의 인자로 전달, Node로 등록하지 않음  |

### 5-4. Conditional Edge를 활용한 RAG 그래프

![image.png]((1)%20LangGraph%EC%9D%98%20%ED%95%B5%EC%8B%AC%20%EA%B5%AC%EC%84%B1%EC%9A%94%EC%86%8C%EC%99%80%20%EA%B8%B0%EB%B3%B8%20RAG%20%ED%8C%A8%ED%84%B4/image%201.png)

```
[START]
  ↓
[retrieve] — 문서 검색
  ↓
[check_doc_relevance] ← 조건부 엣지 함수
  ↓              ↓
관련 있음       관련 없음
  ↓              ↓
[generate]    [rewrite] — 쿼리 재작성
  ↓              ↓
[END]        [retrieve] ← 다시 검색 (루프)
```

```python
# 조건부 엣지 등록 방법
graph_builder.add_conditional_edges("retrieve", check_doc_relevance)
# check_doc_relevance가 "generate" 또는 "rewrite"를 반환 → 자동 라우팅
# 조건부 엣지 함수는 "어디로 갈지"만 결정하고, 실제 작업은 하지 않는다. add_node()로 등록XX
```

---