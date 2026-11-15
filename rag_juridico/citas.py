"""Verificación posterior a la generación: ¿la respuesta se apoya en lo recuperado?

- Citas [F#]: deben corresponder a un fragmento realmente entregado al modelo.
- Menciones de artículos y sentencias: deben aparecer en los fragmentos (o en la propia
  consulta del usuario). Si no, se advierte que podrían provenir del modelo y no del corpus.

No es un evaluador de veracidad: es una alarma barata contra citas inventadas.
"""

from __future__ import annotations

import re

from .bm25 import sin_tildes

RE_CORCHETES = re.compile(r"\[([^\[\]]{1,60})\]")
RE_ID_FUENTE = re.compile(r"\bF\s?(\d{1,2})\b")
RE_ARTICULOS = re.compile(
    r"\bart(?:[íi]culos?|s?\.)\s+((?:\d{1,5}(?:\s*(?:,|y|e|a|al|-)\s*)?)+)", re.IGNORECASE
)
RE_NUMERO = re.compile(r"\d{1,5}")
# Corte Constitucional (C-, T-, SU-) y Corte Suprema, Sala Civil (SC1234-2020), etc.
RE_SENTENCIA = re.compile(r"\b(?:SU|C|T)\s?-\s?\d{1,4}(?!\d)|\bS[CLTP]\s?\d{2,6}\s?-\s?\d{4}\b", re.IGNORECASE)


def extraer_citas(respuesta: str) -> list[int]:
    """Números de fuente citados, en orden de aparición y sin repetir."""
    citados: list[int] = []
    for grupo in RE_CORCHETES.findall(respuesta):
        for numero in RE_ID_FUENTE.findall(grupo):
            if int(numero) not in citados:
                citados.append(int(numero))
    return citados


def _compacto(texto: str) -> str:
    return re.sub(r"\s+", "", sin_tildes(texto.lower()))


def verificar(respuesta: str, textos_fuente: list[str], pregunta: str = "",
              articulos_fuente: list[str] | None = None) -> tuple[list[int], list[str]]:
    """Devuelve (fuentes citadas válidas, advertencias)."""
    advertencias: list[str] = []
    n = len(textos_fuente)
    citados = extraer_citas(respuesta)
    invalidas = [c for c in citados if c < 1 or c > n]
    validas = [c for c in citados if 1 <= c <= n]
    if invalidas:
        advertencias.append(
            "La respuesta cita fuentes que no existen: " + ", ".join(f"[F{c}]" for c in invalidas) + "."
        )
    if not validas:
        advertencias.append("La respuesta no cita ningún fragmento recuperado; revísala con cautela.")

    respaldo = " ".join(textos_fuente + [pregunta])
    numeros_respaldo = set(RE_NUMERO.findall(respaldo)) | set(articulos_fuente or [])
    articulos = []
    for grupo in RE_ARTICULOS.findall(respuesta):
        articulos += RE_NUMERO.findall(grupo)
    sin_respaldo = sorted({a for a in articulos if a not in numeros_respaldo}, key=int)
    if sin_respaldo:
        advertencias.append(
            "Artículos mencionados que no aparecen en los fragmentos recuperados: "
            + ", ".join(sin_respaldo) + "."
        )

    respaldo_compacto = _compacto(respaldo)
    sentencias = {m.group(0).strip() for m in RE_SENTENCIA.finditer(respuesta)}
    faltantes = sorted(s for s in sentencias if _compacto(s) not in respaldo_compacto)
    if faltantes:
        advertencias.append(
            "Sentencias mencionadas que no aparecen en los fragmentos recuperados: "
            + ", ".join(faltantes) + "."
        )
    return validas, advertencias
