import operator
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


class Agent:
    def __init__(self, model, tools):
        self.model = model.bind_tools(tools)

    def __call__(self, state: AgentState):
        messages = state["messages"]
        response = self.model.invoke(messages)
        return {"messages": [response]}


class ChatAgent:
    def __init__(self, model_name="gpt-4o", temperature=0):
        self.model = ChatOpenAI(model_name=model_name, temperature=temperature)

    def chat(self, user_message):
        response = self.model.invoke(user_message)
        return response.content


# Пример использования:
if __name__ == "__main__":
    # Инициализация агента
    chat_agent = ChatAgent()

    # Пример запроса
    user_query = "Привет, как дела?"
    response = chat_agent.chat(user_query)
    print(f"Ответ агента: {response}")

    # Создание графа
    workflow = StateGraph(AgentState)

    # Определение узлов
    # workflow.add_node("agent", Agent(chat_agent.model, [])) # Пример добавления узла агента

    # Определение начальной точки
    # workflow.set_entry_point("agent")

    # Добавление ребра
    # workflow.add_edge("agent", END)

    # Компиляция графа
    # app = workflow.compile()

    # Запуск графа
    # for state in app.stream({"messages": [("user", "Hello")]}):
    #    print(state)
