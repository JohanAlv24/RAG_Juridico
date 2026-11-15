"""Línea de comandos:  python -m rag_juridico <comando>

    ingestar                      indexa los documentos de data/
    estado                        muestra qué hay en el índice
    preguntar "caso..."           responde un caso (o --archivo caso.txt)
    evaluar                       ejecuta los cinco casos de prueba
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rag_juridico", description="RAG jurídico de incumplimiento contractual")
    sub = parser.add_subparsers(dest="comando", required=True)
    sub.add_parser("ingestar", help="Indexa los documentos de data/")
    sub.add_parser("estado", help="Muestra el contenido del índice")
    p_preguntar = sub.add_parser("preguntar", help="Responde un caso o pregunta")
    p_preguntar.add_argument("texto", nargs="?", help="Caso o pregunta entre comillas")
    p_preguntar.add_argument("--archivo", type=Path, help="Archivo de texto con el caso")
    p_preguntar.add_argument("--json", action="store_true", help="Salida en JSON")
    sub.add_parser("evaluar", help="Ejecuta evaluacion/casos_prueba.json")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):  # consola de Windows
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    config = Config.desde_entorno()

    try:
        if args.comando == "ingestar":
            from .ingesta import ingestar

            ingestar(config)
        elif args.comando == "estado":
            from .ingesta import leer_manifiesto

            manifiesto = leer_manifiesto(config)
            print(json.dumps(manifiesto, ensure_ascii=False, indent=2) if manifiesto else "No hay índice todavía.")
        elif args.comando == "preguntar":
            from .pipeline import AsistenteJuridico
            from .presentacion import a_markdown

            texto = args.archivo.read_text(encoding="utf-8") if args.archivo else args.texto
            if not texto:
                parser.error("Indica el caso entre comillas o usa --archivo.")
            respuesta = AsistenteJuridico.desde_storage(config).responder(texto)
            print(json.dumps(respuesta.a_dict(), ensure_ascii=False, indent=2) if args.json else a_markdown(respuesta))
        elif args.comando == "evaluar":
            from .evaluacion import evaluar
            from .pipeline import AsistenteJuridico

            evaluar(AsistenteJuridico.desde_storage(config))
    except RuntimeError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
