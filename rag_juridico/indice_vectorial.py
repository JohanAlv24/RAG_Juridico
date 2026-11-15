"""Índice vectorial con dos implementaciones intercambiables.

- `IndiceQdrant`: Qdrant en modo local persistente (carpeta storage/qdrant). Es el
  predeterminado y el mismo motor usado en clase, pero sin perder el índice al cerrar.
- `IndiceMemoria`: NumPy puro con similitud coseno. Sirve para pruebas y como respaldo
  si Qdrant no está instalado (RAG_BACKEND_VECTORIAL=memoria).

Ambos exponen: recrear(dim), agregar(ids, vectores, tipos), buscar(vector, k), contar().
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class IndiceMemoria:
    def __init__(self, ruta: Path | None = None):
        self.ruta = Path(ruta) if ruta else None
        self.ids: list[str] = []
        self.matriz = np.zeros((0, 0), dtype=np.float32)
        if self.ruta and self.ruta.exists():
            datos = np.load(self.ruta, allow_pickle=False)
            self.matriz = datos["matriz"]
            self.ids = json.loads(str(datos["ids"]))

    def recrear(self, dim: int) -> None:
        self.ids, self.matriz = [], np.zeros((0, dim), dtype=np.float32)

    def agregar(self, ids: list[str], vectores: list[list[float]], tipos: list[str] | None = None) -> None:
        nuevos = np.asarray(vectores, dtype=np.float32)
        normas = np.linalg.norm(nuevos, axis=1, keepdims=True)
        nuevos = nuevos / np.where(normas == 0, 1, normas)
        self.matriz = nuevos if self.matriz.size == 0 else np.vstack([self.matriz, nuevos])
        self.ids.extend(ids)
        if self.ruta:
            self.ruta.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(self.ruta, matriz=self.matriz, ids=json.dumps(self.ids))

    def buscar(self, vector: list[float], k: int) -> list[tuple[str, float]]:
        if not self.ids:
            return []
        consulta = np.asarray(vector, dtype=np.float32)
        norma = np.linalg.norm(consulta)
        puntajes = self.matriz @ (consulta / (norma or 1))
        orden = np.argsort(-puntajes)[:k]
        return [(self.ids[i], float(puntajes[i])) for i in orden]

    def contar(self) -> int:
        return len(self.ids)

    def cerrar(self) -> None:
        pass


class IndiceQdrant:
    def __init__(self, ruta: Path, coleccion: str):
        from qdrant_client import QdrantClient

        Path(ruta).mkdir(parents=True, exist_ok=True)
        self.cliente = QdrantClient(path=str(ruta))
        self.coleccion = coleccion

    def recrear(self, dim: int) -> None:
        from qdrant_client.models import Distance, VectorParams

        if self.cliente.collection_exists(self.coleccion):
            self.cliente.delete_collection(self.coleccion)
        self.cliente.create_collection(
            collection_name=self.coleccion,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

    def agregar(self, ids: list[str], vectores: list[list[float]], tipos: list[str] | None = None) -> None:
        from qdrant_client.models import PointStruct

        tipos = tipos or [""] * len(ids)
        puntos = [
            PointStruct(id=i, vector=[float(x) for x in v], payload={"chunk_id": i, "tipo": t})
            for i, v, t in zip(ids, vectores, tipos)
        ]
        for inicio in range(0, len(puntos), 256):
            self.cliente.upsert(collection_name=self.coleccion, points=puntos[inicio:inicio + 256])

    def buscar(self, vector: list[float], k: int) -> list[tuple[str, float]]:
        respuesta = self.cliente.query_points(
            collection_name=self.coleccion,
            query=[float(x) for x in vector],
            limit=k,
            with_payload=True,
        )
        return [(p.payload["chunk_id"], float(p.score)) for p in respuesta.points]

    def contar(self) -> int:
        if not self.cliente.collection_exists(self.coleccion):
            return 0
        return self.cliente.count(collection_name=self.coleccion, exact=True).count

    def cerrar(self) -> None:
        self.cliente.close()


def abrir_indice(config):
    """Crea el índice según la configuración (qdrant | memoria)."""
    if config.backend_vectorial == "memoria":
        return IndiceMemoria(config.ruta_vectores)
    if config.backend_vectorial == "qdrant":
        try:
            return IndiceQdrant(config.ruta_qdrant, config.coleccion)
        except ImportError as error:
            raise RuntimeError(
                "Falta qdrant-client (pip install qdrant-client) o usa RAG_BACKEND_VECTORIAL=memoria."
            ) from error
    raise ValueError(f"RAG_BACKEND_VECTORIAL desconocido: {config.backend_vectorial!r}")
