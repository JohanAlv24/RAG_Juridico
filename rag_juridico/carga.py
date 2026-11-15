"""Carga de documentos desde data/leyes, data/sentencias y data/casos.

Cada archivo se convierte en un `DocumentoFuente` con su texto por páginas, de modo que
los fragmentos puedan citar la página de la que provienen.

El título y el origen (URL) de cada documento se toman de `data/fuentes.json` si existe:

    {
      "leyes/codigo_civil.pdf": {"titulo": "Código Civil", "origen": "https://..."}
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .config import TIPOS_POR_CARPETA

EXT_TEXTO = {".txt", ".md"}
EXT_MARKITDOWN = {".docx", ".html", ".htm", ".rtf"}
EXTENSIONES = {".pdf"} | EXT_TEXTO | EXT_MARKITDOWN


@dataclass
class DocumentoFuente:
    doc_id: str            # ruta relativa a data/, p. ej. "leyes/codigo_civil.pdf"
    tipo: str              # ley | sentencia | caso
    titulo: str
    archivo: str
    origen: str            # URL o descripción de dónde se obtuvo
    paginas: list[str] = field(default_factory=list)
    paginado: bool = False  # True si los números de página son reales (PDF)

    @property
    def n_caracteres(self) -> int:
        return sum(len(p) for p in self.paginas)


def _leer_texto(ruta: Path) -> str:
    try:
        return ruta.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return ruta.read_text(encoding="latin-1")


def _leer_pdf(ruta: Path) -> list[str]:
    from pypdf import PdfReader

    lector = PdfReader(str(ruta))
    return [(pagina.extract_text() or "") for pagina in lector.pages]


def _leer_markitdown(ruta: Path) -> str:
    try:
        from markitdown import MarkItDown
    except ImportError as error:
        raise RuntimeError(
            f"Para leer {ruta.suffix} instala markitdown: pip install 'markitdown[all]'"
        ) from error
    return MarkItDown(enable_plugins=False).convert(str(ruta)).text_content or ""


def _titulo_por_defecto(ruta: Path) -> str:
    return ruta.stem.replace("_", " ").replace("-", " ").strip().capitalize()


def leer_fuentes_declaradas(dir_datos: Path) -> dict:
    ruta = dir_datos / "fuentes.json"
    if not ruta.exists():
        return {}
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    return {clave: valor for clave, valor in datos.items() if not clave.startswith("_")}


def cargar_documento(ruta: Path, dir_datos: Path, tipo: str, declaradas: dict | None = None) -> DocumentoFuente:
    doc_id = ruta.relative_to(dir_datos).as_posix()
    info = (declaradas or {}).get(doc_id, {})
    extension = ruta.suffix.lower()

    if extension == ".pdf":
        paginas, paginado = _leer_pdf(ruta), True
    elif extension in EXT_TEXTO:
        paginas, paginado = [_leer_texto(ruta)], False
    else:
        paginas, paginado = [_leer_markitdown(ruta)], False

    return DocumentoFuente(
        doc_id=doc_id,
        tipo=tipo,
        titulo=info.get("titulo") or _titulo_por_defecto(ruta),
        archivo=doc_id,
        origen=info.get("origen", ""),
        paginas=paginas,
        paginado=paginado,
    )


def cargar_corpus(dir_datos: Path, verbose: bool = True) -> tuple[list[DocumentoFuente], list[str]]:
    """Devuelve (documentos, avisos). Los avisos explican archivos omitidos o sin texto."""
    dir_datos = Path(dir_datos)
    declaradas = leer_fuentes_declaradas(dir_datos)
    documentos: list[DocumentoFuente] = []
    avisos: list[str] = []

    for carpeta, tipo in TIPOS_POR_CARPETA.items():
        base = dir_datos / carpeta
        if not base.is_dir():
            continue
        for ruta in sorted(base.rglob("*")):
            if not ruta.is_file() or ruta.name.startswith(".") or ruta.name.upper().startswith("LEEME"):
                continue
            if ruta.suffix.lower() not in EXTENSIONES:
                avisos.append(f"Omitido (formato no soportado): {ruta.relative_to(dir_datos).as_posix()}")
                continue
            try:
                doc = cargar_documento(ruta, dir_datos, tipo, declaradas)
            except Exception as error:  # un archivo dañado no debe detener la ingesta
                avisos.append(f"No se pudo leer {ruta.name}: {type(error).__name__}: {error}")
                continue
            if doc.n_caracteres < 200:
                avisos.append(
                    f"{doc.doc_id}: casi no tiene texto extraíble "
                    "(¿PDF escaneado? requiere OCR antes de indexarlo)."
                )
                continue
            if not doc.origen:
                avisos.append(f"{doc.doc_id}: sin origen declarado en data/fuentes.json.")
            documentos.append(doc)
            if verbose:
                print(f"  [{tipo:9}] {doc.titulo}  ({len(doc.paginas)} pág., {doc.n_caracteres:,} caracteres)")

    return documentos, avisos
