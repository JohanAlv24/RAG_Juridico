"""Interfaz gráfica opcional (Gradio).  Uso:  python app.py

Requiere haber ejecutado antes la ingesta:  python -m rag_juridico ingestar
"""

import os

import gradio as gr

from rag_juridico.config import Config
from rag_juridico.ingesta import leer_manifiesto
from rag_juridico.pipeline import AsistenteJuridico, resumir_analisis
from rag_juridico.presentacion import ESTADOS_LEGIBLES, fuentes_markdown
from rag_juridico.prompts import AVISO_LEGAL

config = Config.desde_entorno()
asistente = AsistenteJuridico.desde_storage(config)
manifiesto = leer_manifiesto(config) or {}

EJEMPLOS = [
    ["Compré un apartamento y pagué todo el precio, pero el vendedor no me lo ha entregado tres meses "
     "después de la fecha pactada en la promesa. ¿Qué consecuencias tiene ese incumplimiento?"],
    ["Soy arrendador de un local comercial. El arrendatario lleva tres meses sin pagar el canon y el "
     "contrato tiene una cláusula penal. ¿Qué puedo exigir?"],
    ["Vendí un caballo determinado y, antes de la entrega, murió por un rayo sin culpa mía. "
     "¿Debo devolver el precio o el comprador soporta la pérdida?"],
]


def consultar(caso: str):
    caso = (caso or "").strip()
    if not caso:
        return "Escribe un caso o una pregunta.", "", ""
    try:
        r = asistente.responder(caso)
    except Exception as error:
        return f"**Error:** `{type(error).__name__}: {error}`", "", ""
    respuesta = f"**Estado:** {ESTADOS_LEGIBLES.get(r.estado, r.estado)}\n\n{r.respuesta}"
    if r.advertencias:
        respuesta += "\n\n**⚠ Advertencias de verificación**\n" + "\n".join(f"- {a}" for a in r.advertencias)
    respuesta += f"\n\n{AVISO_LEGAL}"
    analisis = resumir_analisis(r.analisis) + "\n\n**Consultas de búsqueda usadas:**\n" + "\n".join(
        f"- {c[:200]}" for c in r.consultas
    )
    return respuesta, fuentes_markdown(r), analisis


with gr.Blocks(title="RAG jurídico — incumplimiento contractual") as demo:
    gr.Markdown(
        "# RAG jurídico: ¿qué pasa si se incumple el contrato?\n"
        "Describe el caso. El sistema identifica sus elementos, busca en leyes, sentencias y casos "
        "indexados y responde **citando las fuentes**. Si el contexto no alcanza, lo dice."
    )
    with gr.Row():
        with gr.Column(scale=2):
            caja_caso = gr.Textbox(label="Caso o pregunta", lines=7,
                                   placeholder="Partes, tipo de contrato, qué se incumplió, cláusulas pactadas...")
            boton = gr.Button("Analizar caso", variant="primary")
            salida_respuesta = gr.Markdown(label="Respuesta")
        with gr.Column(scale=1):
            gr.Markdown(
                f"**Índice:** {manifiesto.get('fragmentos', '?')} fragmentos "
                f"{manifiesto.get('fragmentos_por_tipo', {})}  \n"
                f"**Embeddings:** `{config.modelo_embeddings}`  \n"
                f"**LLM:** `{config.modelo_generacion}`  \n"
                f"**Recuperación:** híbrida (coseno + BM25), k={config.k_final}, "
                f"umbral de suficiencia={config.umbral_suficiencia}"
            )
            with gr.Accordion("Elementos identificados en el caso", open=False):
                salida_analisis = gr.Markdown()
    gr.Markdown("## Fuentes recuperadas")
    salida_fuentes = gr.Markdown()
    gr.Examples(examples=EJEMPLOS, inputs=caja_caso, label="Casos de ejemplo")

    boton.click(consultar, inputs=caja_caso, outputs=[salida_respuesta, salida_fuentes, salida_analisis])

if __name__ == "__main__":
    # En Colab: RAG_GRADIO_SHARE=1 python app.py  (crea un enlace público temporal)
    demo.queue().launch(show_error=True, share=os.getenv("RAG_GRADIO_SHARE") == "1")
