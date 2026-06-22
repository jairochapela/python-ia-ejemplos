from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from config import API_KEY

MODEL_NAME = "openrouter:qwen/qwen3.7-plus"
model = init_chat_model(
    MODEL_NAME,
    api_key=API_KEY,
    temperature=0.7,
)

system_msg = SystemMessage("""
Eres un asistente muy cuqui, super flower power, que siempre responde con emojis y es muy simpático.
Responde a las preguntas de forma breve y concisa.
""")

# Inicializamos el chat con el mensaje del sistema (system prompt)
messages = [system_msg]

while True:
    user_input = input("Usuario: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    human_msg = HumanMessage(user_input)
    messages.append(human_msg)

    response = model.invoke(messages)
    ai_msg = AIMessage(response.content)
    messages.append(ai_msg)

    print(f"IA: {ai_msg.content}")
