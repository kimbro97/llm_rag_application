# %%
from dotenv import load_dotenv
load_dotenv()
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embedding_function = OpenAIEmbeddings(model = "text-embedding-3-large")

vector_store = Chroma(
    embedding_function = embedding_function,
    collection_name= "income_tax_collection",
    persist_directory= "../chroma/income_tax_collection",
)

retriever = vector_store.as_retriever(search_kwargs = {"k": 3})
# %%
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from langgraph.graph import StateGraph

class AgentState(TypedDict):
    query: str
    context: List[Document]
    answer: str

graph_builder = StateGraph(AgentState)
# %%
def retrieve(state: AgentState) -> AgentState:
    query = state['query']
    docs = retriever.invoke(query)
    return {'context': docs}
# %%
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model = "gpt-4o")
# %%
from langsmith import Client

client = Client()
generate_prompt = client.pull_prompt("rlm/rag-prompt", dangerously_pull_public_prompt=True)

generate_llm = ChatOpenAI(model='gpt-4o', max_completion_tokens=100)

def generate(state: AgentState) -> AgentState:
    context = state['context']
    query = state['query']
    rag_chain = generate_prompt | generate_llm
    response = rag_chain.invoke({'question': query, 'context': context})
    return { 'answer': response.content }
# %%
from langsmith import Client
from typing import Literal
client = Client()

doc_relevance_prompt = client.pull_prompt("langchain-ai/rag-document-relevance", dangerously_pull_public_prompt=True)

def check_doc_relevance(state: AgentState) -> Literal['relevant', 'relevant']:
    context = state['context']
    query = state['query']
    doc_relevance_chain = doc_relevance_prompt | llm
    response = doc_relevance_chain.invoke({'question': query, 'documents': context})
    if response["Score"] == 1:
        return "relevant"
    return "irrelevant"
# %%
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

dictionary = ["사람과 관련된 표현 -> 거주자"]

rewrite_prompt = PromptTemplate.from_template(f"""
사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요
사전: {dictionary}
질문: {{query}}
""")

def rewrite(state: AgentState):
    query = state['query']
    rewrite_chain = rewrite_prompt | llm | StrOutputParser()

    response = rewrite_chain.invoke({'query': query})

    return {"query": response}
# %%
from langchain_core.output_parsers import StrOutputParser

hallucination_prompt = PromptTemplate.from_template("""
당신은 학생의 답변이 문서를 기반으로 하는지 평가하는 교사입니다.
소득세법 발췌문인 문서와 학생의 답변이 주어졌을 때,
학생의 답변이 문서를 기반으로 한다면 "not hallucinated"로 응답하고,
학생의 답변이 문서를 기반으로 하지 않는다면 "hallucinated"로 응답하세요.

documents: {documents}
student_answer: {student_answer}
""")

hallucination_llm = ChatOpenAI(model='gpt-4o', temperature=0)

def check_hallucination(state: AgentState) -> Literal['hallucinated', 'not hallucinated']:
    answer = state['answer']
    context = state['context']
    context = [doc.page_content for doc in context]
    hallucination_chain = hallucination_prompt | hallucination_llm | StrOutputParser()
    response = hallucination_chain.invoke({'student_answer': answer, 'documents': context})

    return response
# %%
helpfulness_prompt = client.pull_prompt("langchain-ai/rag-answer-helpfulness", dangerously_pull_public_prompt=True)

def check_helpfulness_grader(state: AgentState) -> str:

    query = state['query']
    answer = state['answer']

    helpfulness_chain = helpfulness_prompt | llm

    response = helpfulness_chain.invoke({'question': query, 'student_answer': answer})

    if response['Score'] == 1:
        return 'helpful'

    return 'unhelpful'

def check_helpfulness(state: AgentState) -> AgentState:
    return state
# %%
graph_builder.add_node('retrieve', retrieve)
graph_builder.add_node('generate', generate)
graph_builder.add_node('rewrite', rewrite)
graph_builder.add_node('check_helpfulness', check_helpfulness)
# %%
from langgraph.graph import START, END

graph_builder.add_edge(START, 'retrieve')
graph_builder.add_conditional_edges(
    'retrieve',
    check_doc_relevance,
    {
        'relevant': 'generate',
        'irrelevant': END
    }
)
graph_builder.add_conditional_edges(
    'generate',
    check_hallucination,
    {
        'not hallucinated': 'check_helpfulness',
        'hallucinated': 'generate'
    }
)

graph_builder.add_conditional_edges(
    'check_helpfulness',
    check_helpfulness_grader,
    {
        'helpful': END,
        'unhelpful': 'rewrite'
    }
)
graph_builder.add_edge('rewrite', 'retrieve')
# %%
graph = graph_builder.compile()
# %%
from IPython.display import Image, display

display(Image(graph.get_graph().draw_mermaid_png()))
# %%
initial_state = {'query': '연봉 5천만원인 거주자가 납부해야 하는 소득세는 얼마인가요?'}
graph.invoke(initial_state)