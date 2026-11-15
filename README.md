# RAG jurídico: consulta documental sobre incumplimiento contractual

**Proyecto final — SI3003 Inteligencia Artificial**  
**Opción elegida:** Opción A. Sistema RAG  
**Integrantes:** Johan Alexander Álvarez Osorio y Santiago Vera Ramírez

## 1. Problema y alcance

### 1. ¿Qué problema quieren resolver y para quién?

La legislación y la jurisprudencia colombianas contienen información relevante sobre obligaciones e incumplimientos contractuales, pero localizar y comprender las disposiciones pertinentes puede resultar difícil para personas y pequeñas empresas sin formación jurídica. Por lo que se busca facilitar la consulta documental mediante la recuperación de fragmentos de derecho civil y comercial colombiano, acompañados de explicaciones y referencias a las fuentes.

### 2. ¿Qué preguntas o tareas podrá atender el prototipo?

- Identificar los hechos y las obligaciones expresamente mencionados en una consulta contractual.
- Localizar y explicar artículos sobre entrega de bienes, pago de rentas, cláusulas penales, contratos de obra y pérdida fortuita de una cosa.
- Comparar el contenido de disposiciones civiles y mercantiles, señalando sus condiciones y excepciones sin decidir automáticamente cuál rige un caso concreto.
- Indicar cuándo los fragmentos recuperados no permiten ofrecer una explicación fundamentada.

### 3. ¿Por qué eligieron RAG o un agente para ese problema?

Se eligió RAG porque la tarea consiste en consultar y explicar documentos, no en ejecutar acciones externas. El sistema recupera pasajes jurídicos y los proporciona a un modelo de lenguaje para elaborar una respuesta basada en ellos. Esto permite mostrar referencias verificables y controlar el corpus sin reentrenar el modelo. La recuperación no garantiza por sí sola que todas las interpretaciones sean correctas.

### 4. ¿Qué queda por fuera del alcance?

El prototipo no determina responsabilidad contractual, no decide si una demanda prosperaría, no resuelve qué régimen jurídico se aplica en controversias complejas y no calcula indemnizaciones ni valida cláusulas penales concretas. Tampoco sustituye la asesoría jurídica profesional, verifica automáticamente la vigencia de las normas ni cubre materias ajenas al corpus, como el derecho laboral. Sus respuestas deben contrastarse con las fuentes originales.

## 2. Funcionamiento y herramientas

El flujo comprende: extracción y fragmentación de documentos → embeddings e indexación → análisis de la consulta → recuperación híbrida (búsqueda semántica y BM25, fusionadas mediante RRF) → explicación basada en fragmentos → verificación básica de citas. El sistema puede abstenerse cuando no dispone de información adecuada o la consulta queda fuera del alcance definido.

**Fuentes documentales:** Código Civil colombiano ([OEA](https://www.oas.org/dil/esp/codigo_civil_colombia.pdf)); Código de Comercio colombiano, Decreto 410 de 1971 ([Superintendencia de Sociedades](https://www.supersociedades.gov.co/documents/107391/831466/Decreto%2B410%2Bdel%2B27%2Bde%2Bmarzo%2Bde%2B1971.pdf/16152a0c-ab79-5e92-7e13-36326dabfdd6)); y sentencias del corpus, cuyos archivos y orígenes se registran en `data/fuentes.json`.

**Herramientas:** Python, LangChain, NVIDIA NIM, Qdrant local, BM25 y pypdf; Gradio para la interfaz opcional. Se utilizó `nvidia/nemotron-3-embed-1b` para embeddings y, en la evaluación final, `nvidia/nemotron-3.5-lightning-30b-a3b` para generación. Los modelos son configurables.

## 3. Instrucciones de ejecución

**Requisitos:** Python 3.10 o superior, dependencias de `requirements.txt` y una clave de NVIDIA NIM.

1. Instalar las dependencias: `pip install -r requirements.txt`.
2. Crear un archivo `.env` a partir de `.env.example` y configurar `NVIDIA_API_KEY`. Para reproducir la configuración de generación evaluada, establecer también `RAG_MODELO_GENERACION=nvidia/nemotron-3.5-lightning-30b-a3b` y `RAG_MAX_TOKENS=4096`.
3. Ubicar los documentos en `data/leyes/` y `data/sentencias/`, con sus referencias en `data/fuentes.json`.
4. Ejecutar, desde la raíz del proyecto:

```bash
python -m rag_juridico ingestar
python -m rag_juridico estado
python -m rag_juridico preguntar "¿Qué establece el Código Civil sobre la entrega de una cosa vendida?"
python -m rag_juridico evaluar
```

La ingesta se realiza inicialmente y se repite si cambia el corpus. Los cinco casos están definidos en `evaluacion/casos_prueba.json`. 

## 4. Cinco casos de prueba y análisis de resultados


| Caso | Consulta evaluada | Resultado observado | Valoración |
|---|---|---|---|
| **C1** | Entrega tardía de un apartamento y contenido del art. 1882 del Código Civil. | Recupera y explica la obligación de entrega y las alternativas previstas, sin atribuir indemnización concreta. | **Satisfactorio.** |
| **C2** | Falta de pago de rentas y normas sobre cláusula penal. | Explica disposiciones relevantes, pero atribuye erróneamente al Código Civil una referencia al art. 867 del Código de Comercio. | **Parcial.** |
| **C3** | Contrato de obra sin pago del anticipo. | Recupera el art. 2056 y normas sobre perjuicios, pero interpreta indebidamente la facultad de cesar la obra y agrega una norma sobre pagos anticipados poco pertinente. | **Parcial.** |
| **C4** | Comparación de los arts. 1876, 929 y 1729 sobre pérdida de un cuerpo cierto. | Contrasta correctamente las reglas generales y evita decidir el régimen aplicable, aunque simplifica de forma imprecisa una excepción del art. 929. | **Satisfactorio con reserva.** |
| **C5** | Cálculo de indemnización por despido laboral, fuera del corpus. | Se abstiene brevemente, sin calcular cifras ni aplicar normas ajenas al alcance. | **Correcto.** |

**Balance:** los **5/5 estados** coincidieron con los admitidos en las comprobaciones automáticas; esto **no equivale a cinco respuestas jurídicamente correctas**. El tiempo promedio fue de **8,5 segundos por caso**. C1 y C4 muestran utilidad para explicar y comparar fuentes, mientras que C2 y C3 evidencian problemas de fidelidad interpretativa; C5 demuestra el funcionamiento de la abstención por alcance.

**Limitaciones y mejora posible:** la validación automática puede comprobar referencias y estados, pero no demuestra que una fuente respalde fielmente cada conclusión. Además, tres respuestas contractuales omitieron declarar explícitamente la suficiencia y se clasificaron provisionalmente como `parcial`. Como mejoras futuras se plantean una verificación más estricta de la correspondencia entre afirmaciones y fragmentos, y un mejor reordenamiento de las normas recuperadas. El prototipo requiere revisión humana para interpretar sus resultados.
