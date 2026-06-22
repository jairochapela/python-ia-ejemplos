from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from config import API_KEY

MODEL_NAME = "openrouter:deepseek/deepseek-v3.2"
model = init_chat_model(
    MODEL_NAME,
    api_key=API_KEY,
)

SYSTEM_PROMPT = """
Responde a lo que el usuario vaya preguntando citando un proverbio chino inventado en
función del contexto de la pregunta.

El formato de tus respuestas será el siguiente:

Como dice el proverbio chino:
[proverbio en chino]
[traducción]
[conclusión aplicando el proverbio al contexto]
"""

system_msg = SystemMessage(SYSTEM_PROMPT)
human_msg = HumanMessage(input("Formula tu pregunta: "))

messages = [system_msg, human_msg]
response = model.invoke(messages) # Returns AIMessage
print(response.content)