import os
import openai
import typer
from rich.console import Console
from rich.table import Table

"""
Webs de interés:
- Módulo OpenAI: https://github.com/openai/openai-python
- Documentación API ChatGPT: https://platform.openai.com/docs/api-reference/chat
- Typer: https://typer.tiangolo.com
- Rich: https://rich.readthedocs.io/en/stable/
"""

console = Console()

def main():
    # Obtener la clave de la API desde una variable de entorno
    api_key = os.environ.get("OPENAI_API_KEY")

    if api_key is None:
        console.print("[bold red]Error:[/bold red] La clave de la API no está configurada. Configúrala usando la variable de entorno OPENAI_API_KEY.")
        raise typer.Abort()

    openai.api_key = api_key

    console.print("💬 [bold green]ChatGPT API en Python[/bold green]")

    table = Table("Comando", "Descripción")
    table.add_row("exit", "Salir de la aplicación")
    table.add_row("new", "Crear una nueva conversación")

    console.print(table)

    # Contexto del asistente
    context = {"role": "system", "content": "Eres un asistente muy útil."}
    messages = [context]

    while True:
        content = __prompt()

        if content == "new":
            console.print("🆕 Nueva conversación creada")
            messages = [context]
            content = __prompt()

        messages.append({"role": "user", "content": content})

        try:
            # Llamada a la API con encabezado de autorización
            response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            headers={"Authorization": f"Bearer {api_key}"}
            )
            response_content = response.choices[0].message.content

            messages.append({"role": "assistant", "content": response_content})

            console.print(f"[bold green]> [/bold green] [green]{response_content}[/green]")

        except Exception as e:
            console.print("[bold red]Error al comunicarse con la API de OpenAI:[/bold red]")
            console.print(f"[red]{e}[/red]")

def __prompt() -> str:
    prompt = typer.prompt("\n¿Sobre qué quieres hablar? ")

    if prompt == "exit":
        exit = typer.confirm("✋ ¿Estás seguro?")
        if exit:
            console.print("👋 ¡Hasta luego!")
            raise typer.Abort()

        return __prompt()

    return prompt

if __name__ == "__main__":
    typer.run(main)