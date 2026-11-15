"""Búsqueda léxica BM25 (Okapi) en Python puro.

Complementa a los embeddings: en derecho importan las coincidencias exactas
("artículo 1546", "cláusula penal", "SC5185-2021") que la búsqueda semántica puede diluir.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict

STOPWORDS = frozenset("""
a al algo ante con contra cual cuando de del desde donde el ella ellas ellos en entre era
es esa ese eso esta este esto fue ha han hasta la las le les lo los mas me mi muy no nos o
para pero por que se sea ser si sin sobre su sus te tu un una uno unos y ya como e u
""".split())

RE_TOKEN = re.compile(r"[a-z0-9]+")


def sin_tildes(texto: str) -> str:
    descompuesto = unicodedata.normalize("NFD", texto)
    return "".join(c for c in descompuesto if unicodedata.category(c) != "Mn")


def tokenizar(texto: str) -> list[str]:
    tokens = RE_TOKEN.findall(sin_tildes(texto.lower()))
    return [t for t in tokens if t not in STOPWORDS and (len(t) > 1 or t.isdigit())]


class IndiceBM25:
    def __init__(self, textos: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(textos)
        self.longitudes: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, texto in enumerate(textos):
            tokens = tokenizar(texto)
            self.longitudes.append(len(tokens))
            for termino, frecuencia in Counter(tokens).items():
                self.postings[termino].append((i, frecuencia))
        self.longitud_media = (sum(self.longitudes) / self.n) if self.n else 0.0

    def buscar(self, consulta: str, k: int = 10) -> list[tuple[int, float]]:
        """Devuelve [(posición del texto, puntaje)] ordenado de mayor a menor."""
        if not self.n:
            return []
        puntajes: dict[int, float] = defaultdict(float)
        for termino in set(tokenizar(consulta)):
            lista = self.postings.get(termino)
            if not lista:
                continue
            idf = math.log(1 + (self.n - len(lista) + 0.5) / (len(lista) + 0.5))
            for i, tf in lista:
                norma = self.k1 * (1 - self.b + self.b * self.longitudes[i] / (self.longitud_media or 1))
                puntajes[i] += idf * tf * (self.k1 + 1) / (tf + norma)
        return sorted(puntajes.items(), key=lambda par: (-par[1], par[0]))[:k]
