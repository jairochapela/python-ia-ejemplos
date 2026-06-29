import uuid

from pydantic import BaseModel, Field
from datetime import datetime, timedelta
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.agents.middleware import PIIMiddleware
from langgraph.checkpoint.memory import InMemorySaver
import config


class CalendarEvent(BaseModel):
    """
    Evento de calendario con detalles.
    """
    id: str = Field(description="Identificador único del evento")
    title: str = Field(description="Título o denominación del evento")
    start: datetime = Field(description="Fecha y hora de inicio del evento")
    end: datetime = Field(description="Fecha y hora de finalización del evento")

# Almacenamiento de eventos en memoria (simulación de base de datos)
# TODO: Implementar almacenamiento persistente en una base de datos real
AGENDA = []

@tool(parse_docstring=True)
def get_calendar_events(from_day: str, to_day: str) -> list[CalendarEvent]:
    """
    Devuelve los eventos de la agenda entre dos fechas indicadas como parámetros.

    Args:
        from_day (str): Fecha de inicio en formato 'YYYY-MM-DD'.
        to_day (str): Fecha de fin en formato 'YYYY-MM-DD'.

    Returns:
        list[CalendarEvent]: Lista de eventos entre las fechas indicadas.
    """
    from_date = datetime.strptime(from_day, "%Y-%m-%d")
    to_date = datetime.strptime(to_day, "%Y-%m-%d")
    return [event for event in AGENDA if from_date <= event.start <= to_date]

@tool(parse_docstring=True)
def add_calendar_event(title: str, start: str, end: str) -> CalendarEvent:
    """
    Añade un nuevo evento a la agenda con los detalles proporcionados.

    Args:
        title (str): Título o denominación del evento.
        start (str): Fecha y hora de inicio en formato 'YYYY-MM-DD HH:MM'.
        end (str): Fecha y hora de finalización en formato 'YYYY-MM-DD HH:MM'.

    Returns:
        CalendarEvent: El evento añadido a la agenda.
    """
    event = CalendarEvent(
        id=str(len(AGENDA) + 1),
        title=title,
        start=datetime.strptime(start, "%Y-%m-%d %H:%M"),
        end=datetime.strptime(end, "%Y-%m-%d %H:%M")
    )
    AGENDA.append(event)
    return event


@tool(parse_docstring=True)
def relative_date(dias: int = 0, horas: int = 0, minutos: int = 0) -> datetime:
    """
    Devuelve la fecha y hora relativa a la fecha actual según el intervalo indicado como parámetros.
    Se pueden indicar valores positivos o negativos para sumar o restar días, horas y minutos.
    Para obtener la fecha y hora actual, se pueden dejar todos los parámetros en 0.

    Args:
        dias (int): Número de días a sumar o restar.
        horas (int): Número de horas a sumar o restar.
        minutos (int): Número de minutos a sumar o restar.

    Returns:
        datetime: Fecha y hora actual relativa al intervalo indicado.
    """
    return datetime.now() + timedelta(days=dias, hours=horas, minutes=minutos)


llm = init_chat_model(
    "openrouter:openai/gpt-4o-mini",
    api_key=config.API_KEY,
    temperature=0.2,
    max_tokens=256,
    max_retries=1
)

pii_middleware = PIIMiddleware(
    "email",
    strategy="redact",
    apply_to_output=True,
    apply_to_input=True,
)



SYSTEM_PROMPT = """
Eres un asistente inteligente que ayuda a gestionar una agenda de eventos.
Puedes consultar los eventos existentes y añadir nuevos eventos a la agenda, empleando
las herramientas disponibles para ello. Responde de manera clara y concisa, y si no puedes
responder a la pregunta, indica que no tienes la información necesaria.
"""

agente = create_agent(
    model=llm,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_calendar_events, add_calendar_event, relative_date],
    checkpointer=InMemorySaver(),
    middleware=[pii_middleware]
)



thread_id = str(uuid.uuid4())  # Generamos un ID único para la conversación
thread_config = {
    "configurable": {"thread_id": thread_id,}
}

while True:
    entrada_usuario = input(">>")
    if entrada_usuario.lower() in ["salir", "exit", "quit"]:
        print("Saliendo del programa...")
        break

    # Invocamos al agente con la entrada del usuario
    result = agente.invoke({
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

