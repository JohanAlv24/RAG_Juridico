"""Convierte una RespuestaRAG en Markdown para consola, notebook y Gradio."""

from __future__ import annotations

from .pipeline import RespuestaRAG, resumir_analisis
from .prompts import AVISO_LEGAL

ESTADOS_LEGIBLES = {
    "respondida": "✅ Respondida con respaldo en el contexto",
    "parcial": "🟡 Respondida parcialmente (el contexto cubre solo una parte)",
    "insuficiente": "🔴 El modelo declaró que el contexto no es suficiente",
    "sin_contexto": "🔴 Sin contexto suficiente (no se generó respuesta)",
}


def fuentes_markdown(respuesta: RespuestaRAG, con_texto: bool = True, max_caracteres: int = 700) -> str:
    if not respuesta.fuentes:
        return "_No se recuperaron fragmentos suficientemente relacionados._"
    bloques = []
    for f in respuesta.fuentes:
        similitud = f"similitud {f.similitud:.2f}" if f.similitud is not None else "coincidencia léxica"
        marca = "citada" if f.citada else "no citada"
        lineas = [f"**{f.etiqueta}** — {marca}, {similitud}", f"  - Archivo: `{f.archivo}`"]
        if f.origen:
            lineas.append(f"  - Origen: {f.origen}")
        if con_texto:
            texto = f.texto if len(f.texto) <= max_caracteres else f.texto[:max_caracteres].rstrip() + " …"
            lineas.append("  > " + texto.replace("\n", "\n  > "))
        bloques.append("\n".join(lineas))
    return "\n\n".join(bloques)


def a_markdown(respuesta: RespuestaRAG, con_texto: bool = True) -> str:
    partes = [
        f"**Estado:** {ESTADOS_LEGIBLES.get(respuesta.estado, respuesta.estado)}",
        respuesta.respuesta,
    ]
    if respuesta.advertencias:
        partes.append("**⚠ Advertencias de verificación**\n" + "\n".join(f"- {a}" for a in respuesta.advertencias))
    partes.append("---\n### Elementos identificados en el caso\n" + resumir_analisis(respuesta.analisis))
    partes.append("### Fuentes recuperadas\n" + fuentes_markdown(respuesta, con_texto))
    partes.append(AVISO_LEGAL)
    return "\n\n".join(partes)
