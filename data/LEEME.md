# Documentos del corpus

Coloca aquí los documentos que el RAG usará como contexto, según su tipo:

| Carpeta | Qué va | Ejemplos |
|---|---|---|
| `leyes/` | Normas vigentes | Código Civil (Libro IV), Código de Comercio, leyes especiales |
| `sentencias/` | Jurisprudencia | Sentencias de la Corte Suprema (Sala Civil) o de la Corte Constitucional |
| `casos/` | Casos anteriores | Laudos, casos de estudio, conceptos |

Formatos: **PDF con texto seleccionable**, TXT, MD, DOCX, HTML. Un PDF escaneado (imagen) no tiene
texto extraíble: la ingesta lo avisa y lo omite; hay que pasarlo antes por OCR.

Las leyes se dividen por artículo cuando el texto usa encabezados como `ARTÍCULO 1546.` al inicio de línea.

## Origen de cada documento (`fuentes.json`)

El enunciado pide indicar el origen de los documentos. Regístralo en `data/fuentes.json`;
el título y el origen aparecerán junto a cada fuente citada:

```json
{
  "leyes/codigo_civil.pdf": {
    "titulo": "Código Civil colombiano",
    "origen": "URL de donde se descargó"
  }
}
```

Después de agregar o cambiar documentos, vuelve a ejecutar `python -m rag_juridico ingestar`.
