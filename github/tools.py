from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableParallel

llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0)

# Quiz Tool
quiz_prompt_template = "You are a quiz generator. Provide a list of 5 questions based on the user's query.\n\nquery: {query}"
quiz_prompt = ChatPromptTemplate.from_template(quiz_prompt_template)
quiz_chain = quiz_prompt | llm | StrOutputParser()

# Concept Map Tool
concept_map_prompt_template = "You are a concept map generator. Provide a structured concept map based on the user's query.\n\nquery: {query}"
concept_map_prompt = ChatPromptTemplate.from_template(concept_map_prompt_template)
concept_map_chain = concept_map_prompt | llm | StrOutputParser()

# Facts Tool
fact_prompt_template = "You are a bot which generates interesting facts. Provide interesting facts based on the user's query.\n\nquery: {query}"
fact_prompt = ChatPromptTemplate.from_template(fact_prompt_template)
fact_chain = fact_prompt | llm | StrOutputParser()

# Story Tool
story_prompt_template = "You are a creative educational storyteller who provides education in a creative way. Generate a short story based on the user's prompt.\n\nprompt: {prompt}"
story_prompt = ChatPromptTemplate.from_template(story_prompt_template)
story_chain = story_prompt | llm | StrOutputParser()
