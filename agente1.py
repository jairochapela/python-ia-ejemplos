import random

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.tools import tool
import uuid
import config

MODEL_NAME = "openrouter:deepseek/deepseek-v3.2"

# Configuración del modelo LLM
llm = init_chat_model(
    MODEL_NAME,
    api_key=config.API_KEY,
    temperature=0.2
)

# Definición de herramientas para el agente
@tool(parse_docstring=True)
def get_weather(city: str, date: str) -> str:
    """
    Función que devuelve el clima de la ciudad indicada como parámetro.

    Args:
        city (str): Nombre de la ciudad.
        date (str): Fecha para la cual se desea conocer el clima, en formato 'YYYY-MM-DD'.

    Returns:
        str: Clima de la ciudad.
    """
    # Simulamos la obtención del clima.
    opciones_clima = ["soleado", "lluvioso", "nublado", "tormentoso", "nevado"]
    clima = opciones_clima[hash(city + date) % len(opciones_clima)]
    temperatura = random.randint(-5, 35)  # Temperatura aleatoria entre -5 y 35 grados Celsius
    return f"El clima en {city} el {date} es {clima} con una temperatura de {temperatura}°C."


@tool(parse_docstring=True)
def get_current_date() -> str:
    """
    Función que devuelve la fecha actual en formato 'YYYY-MM-DD'.

    Returns:
        str: Fecha actual.
    """
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d")


tools = [get_weather, get_current_date]  # Lista de herramientas (por ahora vacía)


SYSTEM_PROMPT = """
Eres un agente especializado en responder preguntas sobre el clima.
Utiliza las herramientas disponibles para proporcionar información 
precisa y útil.
Escribe tus respuestas de manera clara y concisa, en un tono amigable
y cordial. Si no puedes responder a la pregunta, indica que no tienes 
la información necesaria.
"""


# Inicialización del agente
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT
)

thread_id = str(uuid.uuid4())  # Generamos un ID único para la conversación
thread_config = {
    "configurable": {"thread_id": thread_id,}
}

while True:
    entrada_usuario = input("Pregunta lo que quieras: ")
    if entrada_usuario.lower() in ["salir", "exit", "quit"]:
        print("Saliendo del programa...")
        break

    # Invocamos al agente con la entrada del usuario
    result = agent.invoke({
        "messages": [
            {"role": "user", "content": entrada_usuario}
        ]},
        config=thread_config,        
    )

    for msg in result['messages']:
        match msg.__class__.__name__:
            case "AIMessage":
                print(f"Agente: {msg.content}")
            case "HumanMessage":
                print(f"Tú: {msg.content}")
            case "ToolMessage":
                print(f"Herramienta {msg.name}: {msg.content}")
            case _:
                print(f"Mensaje desconocido: {msg.content}")

