"""Configuración central. Todo valor puede cambiarse con variables de entorno o `.env`."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

try:  # python-dotenv es opcional: sin él solo se leen variables de entorno.
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

RAIZ = Path(__file__).resolve().parent.parent

# Subcarpeta de data/  ->  tipo de fuente jurídica.
TIPOS_POR_CARPETA = {"leyes": "ley", "sentencias": "sentencia"}
ETIQUETAS_TIPO = {"ley": "Ley", "sentencia": "Sentencia"}


def _env(nombre: str, defecto, tipo=str):
    valor = os.getenv(nombre)
    if valor is None or valor.strip() == "":
        return defecto
    return tipo(valor)


@dataclass(frozen=True)
class Config:
    # Rutas
    dir_datos: Path = RAIZ / "data"
    dir_storage: Path = RAIZ / "storage"

    # Modelos (los mismos del curso, NVIDIA NIM)
    modelo_embeddings: str = "nvidia/nemotron-3-embed-1b"
    modelo_generacion: str = "nvidia/nemotron-3.5-lightning-30b-a3b"
    temperatura: float = 0.1
    max_tokens: int = 4096
    modo: str = "nvidia"  # "nvidia" | "simulado" (sin API, solo para probar el flujo)

    # Índice vectorial
    backend_vectorial: str = "qdrant"  # "qdrant" | "memoria"
    coleccion: str = "rag_juridico"

    # Fragmentación
    tam_fragmento: int = 1200
    solape: int = 200

    # Recuperación
    k_denso: int = 12          # candidatos por consulta en la búsqueda semántica
    k_bm25: int = 12           # candidatos por consulta en la búsqueda léxica
    k_final: int = 8           # fragmentos que llegan al LLM
    min_por_tipo: int = 1      # cupo mínimo por tipo de fuente (ley/sentencia/caso)
    umbral_minimo: float = 0.25       # similitud coseno mínima de un fragmento
    umbral_suficiencia: float = 0.35  # si el mejor fragmento no llega aquí, no se responde
    rrf_k: int = 60

    jurisdiccion: str = "Colombia"

    @classmethod
    def desde_entorno(cls, **cambios) -> "Config":
        base = cls()
        cfg = cls(
            dir_datos=Path(_env("RAG_DIR_DATOS", base.dir_datos)),
            dir_storage=Path(_env("RAG_DIR_STORAGE", base.dir_storage)),
            modelo_embeddings=_env("RAG_MODELO_EMBEDDINGS", base.modelo_embeddings),
            modelo_generacion=_env("RAG_MODELO_GENERACION", base.modelo_generacion),
            temperatura=_env("RAG_TEMPERATURA", base.temperatura, float),
            max_tokens=_env("RAG_MAX_TOKENS", base.max_tokens, int),
            modo=_env("RAG_MODO", base.modo).lower(),
            backend_vectorial=_env("RAG_BACKEND_VECTORIAL", base.backend_vectorial).lower(),
            coleccion=_env("RAG_COLECCION", base.coleccion),
            tam_fragmento=_env("RAG_TAM_FRAGMENTO", base.tam_fragmento, int),
            solape=_env("RAG_SOLAPE", base.solape, int),
            k_denso=_env("RAG_K_DENSO", base.k_denso, int),
            k_bm25=_env("RAG_K_BM25", base.k_bm25, int),
            k_final=_env("RAG_K_FINAL", base.k_final, int),
            min_por_tipo=_env("RAG_MIN_POR_TIPO", base.min_por_tipo, int),
            umbral_minimo=_env("RAG_UMBRAL_MINIMO", base.umbral_minimo, float),
            umbral_suficiencia=_env("RAG_UMBRAL_SUFICIENCIA", base.umbral_suficiencia, float),
            rrf_k=_env("RAG_RRF_K", base.rrf_k, int),
            jurisdiccion=_env("RAG_JURISDICCION", base.jurisdiccion),
        )
        return replace(cfg, **cambios) if cambios else cfg

    # Archivos derivados de la ingesta
    @property
    def ruta_fragmentos(self) -> Path:
        return self.dir_storage / "fragmentos.jsonl"

    @property
    def ruta_manifiesto(self) -> Path:
        return self.dir_storage / "manifiesto.json"

    @property
    def ruta_qdrant(self) -> Path:
        return self.dir_storage / "qdrant"

    @property
    def ruta_vectores(self) -> Path:
        return self.dir_storage / "vectores.npz"
