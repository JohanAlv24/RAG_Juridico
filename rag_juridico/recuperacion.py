"""Recuperación híbrida con varias consultas.

1. Cada consulta se busca por significado (embeddings + índice vectorial) y por
   palabras exactas (BM25).
2. Todas las listas se combinan con Reciprocal Rank Fusion (RRF).
3. Se garantiza un cupo mínimo por tipo de fuente (ley / sentencia / caso) cuando existe.
4. Control de suficiencia: si ningún fragmento alcanza `umbral_suficiencia` de similitud
   coseno, el contexto se declara insuficiente y el pipeline no genera respuesta.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .bm25 import IndiceBM25
from .config import Config
from .fragmentacion import Fragmento

MAX_CARACTERES_CONSULTA = 2000


@dataclass
class Resultado:
    fragmento: Fragmento
    score_fusion: float
    score_denso: float | None = None   # similitud coseno máxima entre las consultas
    score_bm25: float | None = None


@dataclass
class Recuperacion:
    resultados: list[Resultado] = field(default_factory=list)
    mejor_score: float = 0.0
    suficiente: bool = False
    n_candidatos: int = 0


def fusion_rrf(listas: list[list[str]], k: int = 60) -> dict[str, float]:
    """RRF: cada lista aporta 1/(k + posición) al puntaje del elemento."""
    puntajes: dict[str, float] = defaultdict(float)
    for lista in listas:
        for posicion, elemento in enumerate(lista, start=1):
            puntajes[elemento] += 1.0 / (k + posicion)
    return dict(puntajes)


def seleccionar_con_cobertura(ordenados: list[Resultado], k: int, min_por_tipo: int) -> list[Resultado]:
    """Toma los k mejores y asegura `min_por_tipo` de cada tipo presente entre los candidatos."""
    seleccion = list(ordenados[:k])
    resto = list(ordenados[k:])
    if min_por_tipo <= 0 or not resto:
        return seleccion
    for tipo in sorted({r.fragmento.tipo for r in ordenados}):
        while sum(r.fragmento.tipo == tipo for r in seleccion) < min_por_tipo:
            candidato = next((r for r in resto if r.fragmento.tipo == tipo), None)
            if candidato is None:
                break
            conteo = Counter(r.fragmento.tipo for r in seleccion)
            # se reemplaza el peor fragmento de un tipo que tenga cupo de sobra
            sobrante = next(
                (r for r in reversed(seleccion) if conteo[r.fragmento.tipo] > min_por_tipo), None
            )
            if sobrante is None:
                break
            seleccion.remove(sobrante)
            seleccion.append(candidato)
            resto.remove(candidato)
    return sorted(seleccion, key=lambda r: -r.score_fusion)


class Recuperador:
    def __init__(self, fragmentos: list[Fragmento], indice, embeddings, config: Config):
        self.config = config
        self.indice = indice
        self.embeddings = embeddings
        self.fragmentos = fragmentos
        self.por_id = {f.chunk_id: f for f in fragmentos}
        self.bm25 = IndiceBM25([f.texto_indexable() for f in fragmentos])

    def recuperar(self, consultas: list[str]) -> Recuperacion:
        cfg = self.config
        consultas = list(dict.fromkeys(c.strip() for c in consultas if c and c.strip()))
        listas: list[list[str]] = []
        denso: dict[str, float] = {}
        lexico: dict[str, float] = {}

        for consulta in consultas:
            vector = self.embeddings.embed_query(consulta[:MAX_CARACTERES_CONSULTA])
            aciertos = [
                (cid, s) for cid, s in self.indice.buscar(vector, cfg.k_denso)
                if s >= cfg.umbral_minimo and cid in self.por_id
            ]
            listas.append([cid for cid, _ in aciertos])
            for cid, s in aciertos:
                denso[cid] = max(s, denso.get(cid, 0.0))

            aciertos_bm25 = [
                (self.fragmentos[i].chunk_id, s) for i, s in self.bm25.buscar(consulta, cfg.k_bm25)
            ]
            listas.append([cid for cid, _ in aciertos_bm25])
            for cid, s in aciertos_bm25:
                lexico[cid] = max(s, lexico.get(cid, 0.0))

        mejor = max(denso.values(), default=0.0)
        suficiente = mejor >= cfg.umbral_suficiencia
        fusion = fusion_rrf(listas, cfg.rrf_k)
        ordenados = [
            Resultado(self.por_id[cid], puntaje, denso.get(cid), lexico.get(cid))
            for cid, puntaje in sorted(fusion.items(), key=lambda par: -par[1])
        ]
        resultados = seleccionar_con_cobertura(ordenados, cfg.k_final, cfg.min_por_tipo) if suficiente else []
        return Recuperacion(resultados, mejor, suficiente, len(ordenados))
