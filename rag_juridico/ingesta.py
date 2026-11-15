"""Ingesta: documentos -> fragmentos -> embeddings -> índice vectorial.

Deja en storage/:
- fragmentos.jsonl : texto y metadatos de cada fragmento (inspeccionable a mano).
- qdrant/ (o vectores.npz) : el índice vectorial.
- manifiesto.json  : qué se indexó, con qué modelo y cuándo.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime

from .carga import cargar_corpus
from .config import Config
from .fragmentacion import Fragmento, fragmentar_corpus
from .indice_vectorial import abrir_indice
from .proveedores import crear_embeddings

LOTE_EMBEDDINGS = 32


class CorpusVacio(RuntimeError):
    pass


def guardar_fragmentos(fragmentos: list[Fragmento], ruta) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", encoding="utf-8") as archivo:
        for fragmento in fragmentos:
            archivo.write(json.dumps(fragmento.a_dict(), ensure_ascii=False) + "\n")


def leer_fragmentos(ruta) -> list[Fragmento]:
    with ruta.open(encoding="utf-8") as archivo:
        return [Fragmento.desde_dict(json.loads(linea)) for linea in archivo if linea.strip()]


def ingestar(config: Config | None = None, embeddings=None, indice=None, verbose: bool = True) -> dict:
    config = config or Config.desde_entorno()
    decir = print if verbose else (lambda *a, **k: None)

    decir(f"1/4 Cargando documentos de {config.dir_datos} ...")
    documentos, avisos = cargar_corpus(config.dir_datos, verbose=verbose)
    for aviso in avisos:
        decir(f"  ⚠ {aviso}")
    if not documentos:
        raise CorpusVacio(
            f"No hay documentos para indexar en {config.dir_datos}. Coloca archivos PDF/TXT/MD/DOCX "
            "en data/leyes y data/sentencias y vuelve a ejecutar la ingesta."
        )

    decir("2/4 Fragmentando ...")
    fragmentos = fragmentar_corpus(documentos, config.tam_fragmento, config.solape)
    por_tipo = Counter(f.tipo for f in fragmentos)
    decir(f"  {len(documentos)} documentos -> {len(fragmentos)} fragmentos {dict(por_tipo)}")

    decir(f"3/4 Generando embeddings ({config.modelo_embeddings if config.modo != 'simulado' else 'simulados'}) ...")
    embeddings = embeddings or crear_embeddings(config)
    textos = [f.texto_indexable() for f in fragmentos]
    vectores: list[list[float]] = []
    for inicio in range(0, len(textos), LOTE_EMBEDDINGS):
        vectores.extend(embeddings.embed_documents(textos[inicio:inicio + LOTE_EMBEDDINGS]))
        decir(f"  {min(inicio + LOTE_EMBEDDINGS, len(textos))}/{len(textos)}", end="\r")
    dimension = len(vectores[0])

    decir(f"\n4/4 Indexando en {config.backend_vectorial} (dimensión {dimension}) ...")
    propio = indice is None
    indice = indice or abrir_indice(config)
    indice.recrear(dimension)
    indice.agregar([f.chunk_id for f in fragmentos], vectores, [f.tipo for f in fragmentos])
    total = indice.contar()
    if propio:
        indice.cerrar()

    guardar_fragmentos(fragmentos, config.ruta_fragmentos)
    manifiesto = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "modo": config.modo,
        "modelo_embeddings": config.modelo_embeddings,
        "dimension": dimension,
        "backend_vectorial": config.backend_vectorial,
        "tam_fragmento": config.tam_fragmento,
        "solape": config.solape,
        "fragmentos": total,
        "fragmentos_por_tipo": dict(por_tipo),
        "documentos": [
            {"archivo": d.archivo, "tipo": d.tipo, "titulo": d.titulo, "origen": d.origen,
             "paginas": len(d.paginas), "fragmentos": sum(f.doc_id == d.doc_id for f in fragmentos)}
            for d in documentos
        ],
        "avisos": avisos,
    }
    config.ruta_manifiesto.write_text(json.dumps(manifiesto, ensure_ascii=False, indent=2), encoding="utf-8")
    decir(f"✅ Índice listo: {total} fragmentos de {len(documentos)} documentos.")
    return manifiesto


def leer_manifiesto(config: Config) -> dict | None:
    if not config.ruta_manifiesto.exists():
        return None
    return json.loads(config.ruta_manifiesto.read_text(encoding="utf-8"))
