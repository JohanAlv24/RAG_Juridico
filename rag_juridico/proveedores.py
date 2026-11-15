"""Modelos de embeddings y de generación (NVIDIA NIM vía LangChain, como en clase).

La clave nunca se escribe en el código. Orden de búsqueda:
secreto de Colab -> variable de entorno / archivo .env -> entrada oculta por teclado.
"""

from __future__ import annotations

import getpass
import os
import re

from .config import Config

RE_PENSAMIENTO = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def obtener_api_key() -> str:
    clave = None
    try:
        from google.colab import userdata  # type: ignore

        clave = userdata.get("NVIDIA_API_KEY")
    except Exception:  # no es Colab o el secreto no existe
        clave = None
    clave = clave or os.getenv("NVIDIA_API_KEY")
    if not clave:
        clave = getpass.getpass("NVIDIA_API_KEY: ").strip()
    if not clave:
        raise ValueError("Falta NVIDIA_API_KEY. Créala en https://build.nvidia.com/")
    os.environ["NVIDIA_API_KEY"] = clave  # las clases de LangChain la leen de aquí
    return clave


def crear_embeddings(config: Config):
    if config.modo == "simulado":
        from .simulado import EmbeddingsSimulados

        return EmbeddingsSimulados()
    obtener_api_key()
    from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

    return NVIDIAEmbeddings(model=config.modelo_embeddings, truncate="END")


def crear_llm(config: Config):
    if config.modo == "simulado":
        from .simulado import LLMSimulado
        return LLMSimulado()

    obtener_api_key()
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    llm = ChatNVIDIA(
        model=config.modelo_generacion,
        temperature=config.temperatura,
        top_p=0.95,
        max_completion_tokens=config.max_tokens,
    )

    return llm.bind(
        chat_template_kwargs={
            "enable_thinking": False
        }
    )


def llamar_llm(llm, sistema: str, usuario: str) -> str:
    """Invoca un chat model de LangChain y devuelve solo el texto de la respuesta."""
    respuesta = llm.invoke([("system", sistema), ("human", usuario)])
    contenido = getattr(respuesta, "content", respuesta)
    if isinstance(contenido, list):  # algunos modelos devuelven bloques de contenido
        contenido = "".join(
            bloque.get("text", "") if isinstance(bloque, dict) else str(bloque) for bloque in contenido
        )
    return RE_PENSAMIENTO.sub("", str(contenido)).strip()
