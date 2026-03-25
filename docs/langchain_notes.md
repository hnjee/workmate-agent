# LangChain 기본

강의 주제: LangChain

## 1. LangChain이란?

- LLM 애플리케이션 개발을 표준화한 프레임워크. 레고 블록처럼 조립해서 복잡한 AI 앱을 만드는 도구.
- LLM 개발 시 반복되는 3가지 문제 해결
    - 모델마다 API 형식이 달라 교체가 어려움 → **통일된 인터페이스**로 해결
    - 프롬프트 하드코딩으로 재사용 불가 → **PromptTemplate**으로 해결
    - 중첩 호출로 코드가 복잡해짐 → LCEL(`|`)로 해결
- **전체적인 흐름: Input → Prompt → LLM → 결과 Parser**

---

## 2. 핵심 개념: LCEL, Runnable

- LLM 활용 흐름: **Input → Prompt → LLM → 결과 Parser**
- 이때 이 모든 과정을 하나하나 호출하여 결과값을 전달하면 복잡해짐 → LCEL는 이 과정을 하나의 체인으로 만들어 준다.
- 각 단계의 모든 컴포넌트(Prompt, LLM, Parser)는 Runnable을 상속받아 `.invoke()` 메서드를 가짐 → `|`로 연결하여 한번에 실행

```python
# 각각 따로 호출하는 대신
prompt_value = prompt.invoke({"text": "hello"})
ai_message = llm.invoke(prompt_value)
result = parser.invoke(ai_message)

# LCEL로 하나의 체인으로 연결
chain = prompt | llm | parser
result = chain.invoke({"text": "hello"})
```

---

## 3. LLM 호출

모든 LLM을 동일한 인터페이스로 사용. 응답은 항상 `AIMessage` 객체로 반환.

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

llm1 = ChatOpenAI(model="gpt-4o")
llm2 = ChatAnthropic(model="claude-3-sonnet")

# 모델이 달라도 동일하게 사용
llm1.invoke("Hello")
llm2.invoke("Hello")
```

> API 키는 항상 `.env` + `dotenv`로 관리. 코드에 하드코딩 금지.
> 

---

## 4. 프롬프트

- 프롬프트 = LLM에게 "뭘 해야 할지" 알려주는 입력
- 프롬프트 작성 팁: 줄바꿈 줄이기 / "Please" 추가 / 출력 형식 명확히 지정
- 랭체인 프롬프트 두 가지 방식

|  | PromptTemplate | ChatPromptTemplate |
| --- | --- | --- |
| 구조 | 단순 문자열 | 역할별 메시지 (system/human/ai) |
| 적합한 경우 | 단순 변환/추출 | 역할/톤/Few-shot 필요할 때 |

### 1) PromptTemplate

```python
# LangChain {}: .invoke() 호출할 때 나중에 채워짐 (재사용 가능)
# 파이썬 문법이 아니라 LangChain이 처리하는 플레이스홀더.
prompt = PromptTemplate.from_template("What is the capital of {country}?")
prompt.invoke({"country": "France"})  # 이때 채워짐
prompt.invoke({"country": "Germany"}) # 같은 템플릿을 다시 사용 가능

# 헷갈릴 수 있는 문법 주의 
# 파이썬 f-string: f 붙여야 하고, 코드 실행 시 즉시 채워짐 (재사용 불가)
name = "France"
f"What is the capital of {name}?"  # 실행 즉시 "What is the capital of France?"

```

### 2) ChatPromptTemplate

### Message 타입

- LangChain은 역할별로 메시지 객체를 제공.
    - **"누가 말하는지" 역할을 구분하는 입력 방식**
    - 역할을 구분하면 LLM이 대화 맥락 파악 쉬움 → 성능 향상

```python
from langchain_core.messages 
import SystemMessage, HumanMessage, AIMessage, ToolMessage

# 1. SystemMessage: AI의 역할/페르소나
SystemMessage(content="You are a professional translator")

# 2. HumanMessage: 사용자 질문
HumanMessage(content="Translate hello to Korean")

# 3. AIMessage: LLM 응답 (Few-shot 예제로 사용)
AIMessage(content="안녕하세요")

# 4. ToolMessage: 도구 실행 결과 (Agent에서 사용)
ToolMessage(content="Temperature: 25°C")
```

### ChatPromptTemplate

```python
# 메시지 형식 (실무에서 주로 사용)
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a professional translator."),
    ("human", "Translate {text} to {language}.")
])
```

- ChatPromptTemplate은 내부적으로 메시지 리스트로 구성된다.
- ChatPromptTemplate로 메세지 객체를 저장하면 변수 주입이 가능하고 LCEL 체이닝이 가능해서 더 랭체인스럽다.
- 단, LangGraph에서는 ChatPromptTemplate를 쓰지 않고 State 안에 `messages: list[BaseMessage]` 형태로  직접 리스트를 만들어서 대화 히스토리를 관리함.

---

## 5. Output Parser

LLM 응답(`AIMessage`)에서 필요한 값을 꺼내는 도구.

### **1) StrOutputParser**

 `AIMessage.content`만 추출 (str 변환이 아님)

```python
from langchain_core.output_parsers import StrOutputParser

chain = prompt | llm | StrOutputParser()  # AIMessage → content(str) 추출
```

### **2) Pydantic + with_structured_output**

구조화된 출력이 필요할 때 권장

```python
from pydantic import BaseModel

class CountryInfo(BaseModel):
    capital: str
    population: int

structured_llm = llm.with_structured_output(CountryInfo)
result = structured_llm.invoke("Tell me about France")
# → CountryInfo(capital="Paris", population=67000000)
```

`JsonOutputParser`는 형식이 불안정해서 사용 금지. Function Calling 기반의 `with_structured_output`을 사용할 것.

---

## 6. LCEL 체이닝

- `|`로 컴포넌트를 연결하는 LangChain의 문법.
- 체인 연결 시 **타입과 키 이름**이 맞아야 함.

```python
# 체인 간 연결 시 형식 변환이 필요한 경우
final_chain = (
    country_chain                        # str "France" 출력
    | {"country": RunnablePassthrough()} # → {"country": "France"} 변환
    | capital_chain                      # dict 입력
)
```

- 체인 연결 패턴 3가지

| 패턴 | 용도 |
| --- | --- |
| 체인 직접 사용 `{"key": some_chain}` | 가장 간결 |
| lambda `{"key": lambda x: x}` | 가공이 필요할 때 |
| RunnablePassthrough | 그대로 전달, 명시적 |
- 추가 기능: `.stream()` / `.batch()` / `.ainvoke()`

---

## 7. 베스트 프랙티스: 체인 분리

> "LLM은 하라고 훈련됨. 하지 마라는 잘 안 먹힘."
→ 안전성 체크, 라우팅, 분기는 별도 체인으로 분리할 것.
> 

```python
# ❌ 한 프롬프트에 다 넣기
"Answer the question. But DO NOT answer if it's about violence..."

# ✅ 체인으로 분리
safety_chain = safety_prompt | small_llm | with_structured_output(SafetyCheck)
answer_chain = answer_prompt | big_llm | StrOutputParser()

if safety_chain.invoke(...).is_safe:
    return answer_chain.invoke(...)
```

> 이 체인 분리 철학은 LangGraph의 노드 분리로 이어짐.
>