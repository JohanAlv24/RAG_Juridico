"""Ejecuta los casos de prueba de evaluacion/casos_prueba.json y genera el informe.

La valoración es MANUAL (como pide el enunciado): el informe deja un espacio para escribirla.
Las comprobaciones automáticas son solo una ayuda para la revisión:
- estado: ¿el sistema respondió / se abstuvo como se esperaba?
- tipos de fuente: ¿recuperó leyes, sentencias o casos según lo esperado?
- términos: ¿la respuesta menciona los conceptos clave esperados?
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .bm25 import sin_tildes
from .config import RAIZ
from .pipeline import AsistenteJuridico, RespuestaRAG
from .presentacion import ESTADOS_LEGIBLES, fuentes_markdown

RUTA_CASOS = RAIZ / "evaluacion" / "casos_prueba.json"
DIR_RESULTADOS = RAIZ / "evaluacion" / "resultados"


def comprobar(caso: dict, respuesta: RespuestaRAG) -> dict:
    texto = sin_tildes(respuesta.respuesta.lower())
    tipos = {f.tipo for f in respuesta.fuentes}
    esperados = caso.get("estados_esperados") or []
    terminos = caso.get("terminos_esperados") or []
    return {
        "estado_ok": (respuesta.estado in esperados) if esperados else None,
        "tipos_faltantes": sorted(set(caso.get("tipos_fuente_esperados") or []) - tipos),
        "terminos_faltantes": [t for t in terminos if sin_tildes(t.lower()) not in texto],
    }


def evaluar(asistente: AsistenteJuridico, ruta_casos: Path = RUTA_CASOS,
            dir_resultados: Path = DIR_RESULTADOS, verbose: bool = True) -> Path:
    casos = json.loads(Path(ruta_casos).read_text(encoding="utf-8"))["casos"]
    marca = datetime.now().strftime("%Y%m%d_%H%M%S")
    lineas = [f"# Resultados de evaluación — {marca}", ""]
    registros = []

    for caso in casos:
        if verbose:
            print(f"▶ {caso['id']} ({caso['categoria']}) ...")
        respuesta = asistente.responder(caso["pregunta"])
        chequeo = comprobar(caso, respuesta)
        registros.append({"caso": caso, "respuesta": respuesta.a_dict(), "comprobaciones": chequeo})

        estado_ok = {True: "sí", False: "NO", None: "n/a"}[chequeo["estado_ok"]]
        lineas += [
            f"## {caso['id']} — {caso['categoria']}",
            "", f"**Pregunta / caso:** {caso['pregunta']}",
            "", f"**Resultado esperado:** {caso['resultado_esperado']}",
            "", f"**Estado obtenido:** {ESTADOS_LEGIBLES.get(respuesta.estado, respuesta.estado)} "
                f"(mejor similitud {respuesta.mejor_similitud:.2f}, {respuesta.segundos:.1f} s)",
            "", "**Respuesta obtenida:**", "", respuesta.respuesta,
            "", "**Fuentes recuperadas:**", "", fuentes_markdown(respuesta, con_texto=False),
            "", "**Comprobaciones automáticas (ayuda, no reemplazan la revisión manual):**",
            f"- ¿Estado esperado? {estado_ok} (esperado: {', '.join(caso.get('estados_esperados') or ['—'])})",
            f"- Tipos de fuente esperados que faltaron: {', '.join(chequeo['tipos_faltantes']) or 'ninguno'}",
            f"- Términos esperados ausentes: {', '.join(chequeo['terminos_faltantes']) or 'ninguno'}",
            f"- Advertencias de citas: {'; '.join(respuesta.advertencias) or 'ninguna'}",
            "", "**Valoración manual del equipo:** _(¿es relevante?, ¿cada afirmación está respaldada "
                "por el fragmento citado?, ¿coincide con el documento original?)_",
            "", "---", "",
        ]

    dir_resultados.mkdir(parents=True, exist_ok=True)
    ruta_md = dir_resultados / f"resultados_{marca}.md"
    ruta_md.write_text("\n".join(lineas), encoding="utf-8")
    (dir_resultados / f"resultados_{marca}.json").write_text(
        json.dumps(registros, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if verbose:
        print(f"✅ Informe: {ruta_md}")
    return ruta_md
