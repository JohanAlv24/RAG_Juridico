"""Modelos simulados (RAG_MODO=simulado): no usan red ni clave.

Sirven únicamente para comprobar que la instalación y el flujo completo funcionan
(ingesta, recuperación, fuentes, abstención). Sus respuestas NO son análisis jurídico.
"""

from __future__ import annotations

import hashlib
import json
import math

from .bm25 import tokenizar

DIMENSION = 384


class EmbeddingsSimulados:
    """Bolsa de palabras con hashing: textos con vocabulario parecido quedan cerca."""

    def _vector(self, texto: str) -> list[float]:
        vector = [0.0] * DIMENSION
        for token in tokenizar(texto):
            posicion = int(hashlib.md5(token.encode()).hexdigest(), 16) % DIMENSION
            vector[posicion] += 1.0
        norma = math.sqrt(sum(x * x for x in vector)) or 1.0
        return [x / norma for x in vector]

    def embed_documents(self, textos: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in textos]

    def embed_query(self, texto: str) -> list[float]:
        return self._vector(texto)


class _Mensaje:
    def __init__(self, content: str):
        self.content = content


class LLMSimulado:
    def invoke(self, mensajes):
        sistema, usuario = mensajes[0][1], mensajes[-1][1]
        if "JSON" in sistema:  # etapa de análisis del caso
            return _Mensaje(json.dumps({
                "tipo_contrato": "no determinado (modo simulado)",
                "partes": [], "obligaciones": [], "incumplimiento": "",
                "causa": "no_determinada", "clausulas_relevantes": [],
                "pregunta_juridica": usuario[-300:], "datos_faltantes": [],
                "consultas_busqueda": [],
            }, ensure_ascii=False))
        return _Mensaje(
            "SUFICIENCIA: PARCIAL\n"
            "**[MODO SIMULADO]** Esta respuesta no la generó un modelo de lenguaje. "
            "El flujo funcionó y el fragmento más relevante recuperado fue [F1]. "
            "Configura NVIDIA_API_KEY y quita RAG_MODO=simulado para obtener el análisis real."
        )
