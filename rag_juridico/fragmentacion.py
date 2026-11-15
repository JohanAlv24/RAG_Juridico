"""Fragmentación (chunking) consciente de la estructura jurídica.

- Leyes: un fragmento por artículo, para poder citar "Código Civil, art. 1546".
  Los artículos muy largos se subdividen conservando el número de artículo.
- Sentencias y casos: división recursiva (párrafo -> línea -> oración -> palabra) con
  solapamiento, anotando la sección (ANTECEDENTES, CONSIDERACIONES, RESUELVE...).

Cada fragmento conserva tipo, título, archivo, origen, página, artículo y sección.
"""

from __future__ import annotations

import re
import uuid
from bisect import bisect_right
from dataclasses import asdict, dataclass

from .carga import DocumentoFuente
from .config import ETIQUETAS_TIPO

SEPARADORES = ["\n\n", "\n", ". ", "; ", " "]
MIN_CARACTERES = 40

# "ARTÍCULO 1546.", "Artículo 2o.", "ART. 870" al inicio de una línea (no en minúscula,
# para no confundir remisiones dentro de un párrafo con el inicio de un artículo).
RE_ARTICULO = re.compile(
    r"(?m)^[ \t]*(?:ART[ÍI]CULO|Art[íi]culo|ART\.|Art\.)[ \t]+(\d+(?:[.\-]\d+)*)[ \t]*(?:[º°]|o\.)?"
)

RE_SECCION = re.compile(
    r"(?m)^[ \t]*(?:[IVXLC]+[.\-)]|\d+[.\-)])?[ \t]*"
    r"((?:ANTECEDENTES|HECHOS|DEMANDA|PRETENSIONES|CARGOS|PROBLEMA JUR[ÍI]DICO|CONSIDERACIONES|"
    r"FUNDAMENTOS|CASO CONCRETO|DECISI[ÓO]N|RESUELVE|SALVAMENTO DE VOTO|ACLARACI[ÓO]N DE VOTO)"
    r"[^\n]{0,60})$"
)


@dataclass
class Fragmento:
    chunk_id: str
    doc_id: str
    tipo: str
    titulo: str
    archivo: str
    origen: str
    indice: int
    texto: str
    pagina: int | None = None
    articulo: str | None = None
    seccion: str | None = None

    def localizador(self) -> str:
        partes = []
        if self.articulo:
            partes.append(f"art. {self.articulo}")
        if self.seccion:
            partes.append(self.seccion)
        if self.pagina:
            partes.append(f"p. {self.pagina}")
        return ", ".join(partes) or f"fragmento {self.indice + 1}"

    def encabezado(self) -> str:
        return f"{ETIQUETAS_TIPO.get(self.tipo, self.tipo)} | {self.titulo} | {self.localizador()}"

    def texto_indexable(self) -> str:
        """Texto que se convierte en embedding: el encabezado da contexto al fragmento."""
        return f"{self.encabezado()}\n{self.texto}"

    def a_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def desde_dict(cls, datos: dict) -> "Fragmento":
        return cls(**datos)


# ---------------------------------------------------------------- normalización

def normalizar(texto: str) -> str:
    texto = texto.replace("\r\n", "\n").replace("\r", "\n").replace("­", "")
    texto = re.sub(r"(\w)-\n(\w)", r"\1\2", texto)      # palabras cortadas al final de línea
    texto = re.sub(r"[ \t ]+", " ", texto)
    texto = re.sub(r" ?\n ?", "\n", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return texto.strip()


def unir_paginas(paginas: list[str]) -> tuple[str, list[int]]:
    """Une las páginas y devuelve las posiciones donde empieza cada una."""
    partes, inicios, posicion = [], [], 0
    for pagina in paginas:
        limpia = normalizar(pagina)
        inicios.append(posicion)
        partes.append(limpia)
        posicion += len(limpia) + 2
    return "\n\n".join(partes), inicios


# ---------------------------------------------------------- división recursiva

def _atomizar(texto: str, ini: int, fin: int, tam: int, nivel: int = 0) -> list[tuple[int, int]]:
    """Parte texto[ini:fin] en tramos contiguos de longitud <= tam, por el separador más natural."""
    if fin - ini <= tam:
        return [(ini, fin)]
    if nivel >= len(SEPARADORES):
        return [(i, min(i + tam, fin)) for i in range(ini, fin, tam)]
    separador = SEPARADORES[nivel]
    tramos, pos = [], ini
    while True:
        corte = texto.find(separador, pos, fin)
        if corte == -1:
            break
        corte += len(separador)
        tramos.append((pos, corte))
        pos = corte
    if pos < fin:
        tramos.append((pos, fin))
    if len(tramos) <= 1:
        return _atomizar(texto, ini, fin, tam, nivel + 1)
    resultado = []
    for a, b in tramos:
        resultado.extend(_atomizar(texto, a, b, tam, nivel + 1) if b - a > tam else [(a, b)])
    return resultado


def dividir_texto(texto: str, tam: int, solape: int, ini: int = 0, fin: int | None = None) -> list[tuple[int, int]]:
    """Devuelve tramos (inicio, fin) de hasta `tam` caracteres con ~`solape` de traslape."""
    fin = len(texto) if fin is None else fin
    atomos = _atomizar(texto, ini, fin, tam)
    tramos: list[tuple[int, int]] = []
    actual: list[tuple[int, int]] = []
    for atomo in atomos:
        if actual and atomo[1] - actual[0][0] > tam:
            tramos.append((actual[0][0], actual[-1][1]))
            cola: list[tuple[int, int]] = []
            for previo in reversed(actual):
                if actual[-1][1] - previo[0] <= solape and atomo[1] - previo[0] <= tam:
                    cola.insert(0, previo)
                else:
                    break
            actual = cola
        actual.append(atomo)
    if actual:
        tramos.append((actual[0][0], actual[-1][1]))
    return tramos


# ------------------------------------------------------------------ fragmentar

def _tramos_por_articulo(texto: str, tam: int, solape: int) -> list[tuple[int, int, str | None]] | None:
    coincidencias = list(RE_ARTICULO.finditer(texto))
    if len(coincidencias) < 3:
        return None  # no parece un texto articulado: se usa la división recursiva
    tramos: list[tuple[int, int, str | None]] = []
    if coincidencias[0].start() > 0:  # preámbulo antes del primer artículo
        tramos += [(a, b, None) for a, b in dividir_texto(texto, tam, solape, 0, coincidencias[0].start())]
    for i, m in enumerate(coincidencias):
        ini = m.start()
        fin = coincidencias[i + 1].start() if i + 1 < len(coincidencias) else len(texto)
        numero = m.group(1).rstrip(".-")
        if fin - ini <= int(tam * 1.5):
            tramos.append((ini, fin, numero))
        else:
            tramos += [(a, b, numero) for a, b in dividir_texto(texto, tam, solape, ini, fin)]
    return tramos


def fragmentar_documento(doc: DocumentoFuente, tam: int = 1200, solape: int = 200) -> list[Fragmento]:
    texto, inicios_pagina = unir_paginas(doc.paginas)
    if not texto:
        return []

    tramos = _tramos_por_articulo(texto, tam, solape) if doc.tipo == "ley" else None
    if tramos is None:
        tramos = [(a, b, None) for a, b in dividir_texto(texto, tam, solape)]

    secciones = [(m.start(), m.group(1).strip(" .:")) for m in RE_SECCION.finditer(texto)] \
        if doc.tipo != "ley" else []
    inicios_seccion = [pos for pos, _ in secciones]

    fragmentos: list[Fragmento] = []
    for ini, fin, articulo in tramos:
        contenido = texto[ini:fin].strip()
        if len(contenido) < MIN_CARACTERES and len(tramos) > 1 and not articulo:
            continue  # restos sin contenido útil (los artículos cortos sí se conservan)
        seccion = None
        if secciones:
            # sección vigente en la mitad del fragmento (el título puede venir dentro de él)
            j = bisect_right(inicios_seccion, (ini + fin) // 2) - 1
            seccion = secciones[j][1].title() if j >= 0 else None
        indice = len(fragmentos)
        fragmentos.append(Fragmento(
            chunk_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc.doc_id}#{indice}")),
            doc_id=doc.doc_id,
            tipo=doc.tipo,
            titulo=doc.titulo,
            archivo=doc.archivo,
            origen=doc.origen,
            indice=indice,
            texto=contenido,
            pagina=(bisect_right(inicios_pagina, ini) if doc.paginado else None),
            articulo=articulo,
            seccion=seccion,
        ))
    return fragmentos


def fragmentar_corpus(documentos: list[DocumentoFuente], tam: int = 1200, solape: int = 200) -> list[Fragmento]:
    fragmentos: list[Fragmento] = []
    for doc in documentos:
        fragmentos.extend(fragmentar_documento(doc, tam, solape))
    return fragmentos
