"""RAG jurídico: consecuencias del incumplimiento de un contrato.

Flujo general (ver README.md):

    documentos -> carga -> fragmentos -> embeddings -> índice vectorial   (ingesta)
    caso -> análisis -> recuperación híbrida -> control de suficiencia
         -> generación con citas -> verificación de citas                (consulta)
"""

from .config import Config

__all__ = ["Config"]
__version__ = "1.0.0"
