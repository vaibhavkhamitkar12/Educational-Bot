from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from tools import quiz_chain, concept_map_chain, fact_chain, story_chain
from retriever import retriever
from langgraph.graph import StateGraph, END
from operator import itemgetter
from langchain_groq import ChatGroq

# LLM initialization (adjust accordingly)
llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0)

# Router logic
router_prompt_template = """
You are an expert in routing user queries to appropriate tools.
If the user greets you, route to 'llm_fallback'.
If the user asks for information retrieval, route to the 'VectorStore'.
If the user asks for generating a quiz, route to the 'QuizGenerator'.
If the user asks for creating a concept map, route to the 'ConceptMapGenerator'.
If the user asks for facts, route to 'Facts'.
If the user asks for a story, route to 'Story'.
Otherwise, output 'llm_fallback'.
query: {query}
"""
prompt = ChatPromptTemplate.from_template(router_prompt_template)
question_router = prompt | llm.bind_tools(tools=[retriever.sim_search, quiz_chain.invoke])

def question_router_node(state):
    query = state.query
    try:
        response = question_router.invoke({"query": query})
        tool_name = response.additional_kwargs["tool_calls"][0]["name"]
    except Exception as e:
        return "llm_fallback"
    return tool_name

# Create the workflow
workflow = StateGraph()

workflow.add_node("VectorStore", retriever.sim_search)
workflow.add_node("QuizGenerator", lambda state: quiz_chain.invoke({"query": state.query}))
workflow.add_node("ConceptMapGenerator", lambda state: concept_map_chain.invoke({"query": state.query}))
workflow.add_node("Facts", lambda state: fact_chain.invoke({"query": state.query}))
workflow.add_node("Story", lambda state: story_chain.invoke({"query": state.query}))

workflow.set_conditional_entry_point(
    question_router_node,
    {
        "llm_fallback": "fallback",
        "VectorStore": "VectorStore",
        "QuizGenerator": "QuizGenerator",
        "ConceptMapGenerator": "ConceptMapGenerator",
        "Facts": "Facts",
        "Story": "Story",
    },
)

workflow.add_edge("VectorStore", "rag")

app = workflow.compile(debug=False)
