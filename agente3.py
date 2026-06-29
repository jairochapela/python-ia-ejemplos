from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
import config
import ast
import uuid

SYSTEM_PROMPT = """
Eres un asistente inteligente que responde a preguntas y puede usar herramientas para
realizar cálculos matemáticos.
Si la pregunta requiere evaluar expresiones matemáticas, llama a la herramienta
adecuada con los parámetros necesarios.
"""

@tool(parse_docstring=True)
def calcular(expresion: str) -> float:
    """
    Función que evalúa una expresión matemática y devuelve el resultado.

    Args:
        expresion (str): Expresión matemática a evaluar.

    Returns:
        float: Resultado de la evaluación de la expresión.
    """
    def evaluar_nodo(nodo: ast.AST) -> float:
        if isinstance(nodo, ast.Constant):
            return nodo.value
        elif isinstance(nodo, ast.BinOp):
            izquierda = evaluar_nodo(nodo.left)
            derecha = evaluar_nodo(nodo.right)
            if isinstance(nodo.op, ast.Add):
                return izquierda + derecha
            elif isinstance(nodo.op, ast.Sub):
                return izquierda - derecha
            elif isinstance(nodo.op, ast.Mult):
                return izquierda * derecha
            elif isinstance(nodo.op, ast.Div):
                return izquierda / derecha
            else:
                raise ValueError("Operación no soportada")
        else:
            raise ValueError("Expresión no válida")
        
    # Primero compilamos la expresión en un árbol de sintaxis abstracta (AST)
    nodo = ast.parse(expresion, mode='eval')
    # Luego evaluamos el nodo raíz del AST
    resultado = evaluar_nodo(nodo.body)
    return resultado


llm = init_chat_model(
    "openrouter:openai/gpt-4o-mini",
    api_key=config.API_KEY,
    temperature=0.2
)

# Lista de herramientas disponibles para el agente
tools = [calcular] 

agente = create_agent(
    model=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
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

