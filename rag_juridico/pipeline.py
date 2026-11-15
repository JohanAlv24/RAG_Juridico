"""Pipeline de consulta del RAG jurídico.

    caso ──► 1. analizar_caso ──► 2. recuperar ──► 3. ¿contexto suficiente? ──no──► abstención
                                                          │ sí
                                                          ▼
                                    4. generar con citas ──► 5. verificar citas ──► respuesta + fuentes
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field

from . import prompts
from .citas import verificar
from .config import Config, ETIQUETAS_TIPO
from .ingesta import leer_fragmentos, leer_manifiesto
from .indice_vectorial import abrir_indice
from .proveedores import crear_embeddings, crear_llm, llamar_llm
from .recuperacion import Recuperador, Resultado

RE_SUFICIENCIA = re.compile(r"^\W*SUFICIENCIA\W*(SUFICIENTE|PARCIAL|INSUFICIENTE)\W*$", re.IGNORECASE | re.MULTILINE)
ESTADOS = {"SUFICIENTE": "respondida", "PARCIAL": "parcial", "INSUFICIENTE": "insuficiente"}


@dataclass
class Fuente:
    numero: int            # el N de [FN]
    tipo: str
    titulo: str
    localizador: str       # art. / sección / página
    archivo: str
    origen: str
    texto: str
    similitud: float | None
    citada: bool = False

    @property
    def etiqueta(self) -> str:
        return f"[F{self.numero}] {ETIQUETAS_TIPO.get(self.tipo, self.tipo)} · {self.titulo} · {self.localizador}"


@dataclass
class RespuestaRAG:
    pregunta: str
    estado: str                      # respondida | parcial | insuficiente | sin_contexto
    respuesta: str
    analisis: dict | None = None
    consultas: list[str] = field(default_factory=list)
    fuentes: list[Fuente] = field(default_factory=list)
    advertencias: list[str] = field(default_factory=list)
    mejor_similitud: float = 0.0
    segundos: float = 0.0

    def a_dict(self) -> dict:
        return asdict(self)


def extraer_json(texto: str) -> dict | None:
    """Obtiene el primer objeto JSON de la salida del modelo (tolera ```json ... ``` y texto extra)."""
    inicio, fin = texto.find("{"), texto.rfind("}")
    if inicio == -1 or fin <= inicio:
        return None
    try:
        datos = json.loads(texto[inicio:fin + 1])
    except json.JSONDecodeError:
        return None
    return datos if isinstance(datos, dict) else None


def resumir_analisis(analisis: dict | None) -> str:
    if not analisis:
        return "(no se pudo estructurar el caso; se usa el texto original)"
    lineas = []
    for clave, titulo in [
        ("tipo_contrato", "Tipo de contrato"), ("partes", "Partes"), ("obligaciones", "Obligaciones"),
        ("incumplimiento", "Incumplimiento"), ("causa", "Causa"),
        ("clausulas_relevantes", "Cláusulas relevantes"), ("pregunta_juridica", "Pregunta jurídica"),
        ("datos_faltantes", "Datos que faltan"),
    ]:
        valor = analisis.get(clave)
        if not valor:
            continue
        if isinstance(valor, list):
            valor = "; ".join(
                (f"{v.get('nombre', '?')} ({v.get('rol', '?')})" if isinstance(v, dict) else str(v)) for v in valor
            )
        lineas.append(f"- {titulo}: {valor}")
    return "\n".join(lineas) or "(sin elementos identificados)"


def formatear_contexto(resultados: list[Resultado]) -> str:
    return "\n\n---\n\n".join(
        f"[F{i}] {r.fragmento.encabezado()}\n{r.fragmento.texto}" for i, r in enumerate(resultados, start=1)
    )


class AsistenteJuridico:
    def __init__(self, config: Config, embeddings, llm, indice, fragmentos):
        self.config = config
        self.llm = llm
        self.indice = indice
        self.fragmentos = fragmentos
        self.recuperador = Recuperador(fragmentos, indice, embeddings, config)

    @classmethod
    def desde_storage(cls, config: Config | None = None) -> "AsistenteJuridico":
        """Carga el índice creado por la ingesta (`python -m rag_juridico ingestar`)."""
        config = config or Config.desde_entorno()
        manifiesto = leer_manifiesto(config)
        if manifiesto is None or not config.ruta_fragmentos.exists():
            raise RuntimeError("No hay índice. Ejecuta primero: python -m rag_juridico ingestar")
        if manifiesto.get("modo") != config.modo or (
            config.modo != "simulado" and manifiesto.get("modelo_embeddings") != config.modelo_embeddings
        ):
            raise RuntimeError(
                "El índice se creó con otro modelo de embeddings o en otro modo "
                f"({manifiesto.get('modo')} / {manifiesto.get('modelo_embeddings')}). Vuelve a ejecutar la ingesta."
            )
        return cls(
            config,
            crear_embeddings(config),
            crear_llm(config),
            abrir_indice(config),
            leer_fragmentos(config.ruta_fragmentos),
        )

    # ------------------------------------------------------------------ etapas

    def analizar_caso(self, pregunta: str) -> tuple[dict | None, list[str]]:
        """Etapa 1: elementos del caso y consultas de búsqueda en lenguaje jurídico."""
        consultas = [pregunta]
        try:
            salida = llamar_llm(
                self.llm, prompts.SISTEMA_ANALISIS.format(jurisdiccion=self.config.jurisdiccion), pregunta
            )
            analisis = extraer_json(salida)
        except Exception:  # si el análisis falla, se sigue con la pregunta original
            analisis = None
        if analisis:
            extras = analisis.get("consultas_busqueda") or []
            consultas += [str(c) for c in extras if isinstance(c, str) and c.strip()][:4]
            if analisis.get("pregunta_juridica"):
                consultas.append(str(analisis["pregunta_juridica"]))
        return analisis, list(dict.fromkeys(consultas))

    def responder(self, pregunta: str) -> RespuestaRAG:
        inicio = time.time()
        pregunta = (pregunta or "").strip()
        if not pregunta:
            return RespuestaRAG(pregunta, "sin_contexto", "Escribe el caso o la pregunta que quieres consultar.")


        analisis, consultas = self.analizar_caso(pregunta)

        # Comprobar si la consulta está fuera del alcance documental
        tipo_contrato = str(
            (analisis or {}).get("tipo_contrato") or ""
        ).strip().lower()

        consulta_laboral = (
            tipo_contrato in {
                "laboral",
                "contrato laboral",
                "contrato de trabajo"
            }
            or bool(
                re.search(
                    r"\b(contrato laboral|despido|despid\w*)\b",
                    pregunta,
                    flags=re.IGNORECASE
                )
            )
        )

        if consulta_laboral:
            return RespuestaRAG(
                pregunta=pregunta,
                estado="insuficiente",
                respuesta=(
                    "Esta consulta corresponde al derecho laboral, "
                    "una materia fuera del alcance de este prototipo "
                    "de consulta documental sobre contratos civiles "
                    "y comerciales. No es posible responderla "
                    "adecuadamente con el corpus disponible."
                ),
                analisis=analisis,
                consultas=consultas,
                segundos=time.time() - inicio,
            )


        recuperacion = self.recuperador.recuperar(consultas)

        # Etapa 3: abstención determinista, sin llamar al modelo generador.
        if not recuperacion.suficiente:
            return RespuestaRAG(
                pregunta, "sin_contexto",
                prompts.MENSAJE_SIN_CONTEXTO.format(
                    mejor=recuperacion.mejor_score, umbral=self.config.umbral_suficiencia
                ),
                analisis, consultas, [], [], recuperacion.mejor_score, time.time() - inicio,
            )

        # Etapa 4: generación restringida al contexto.
        salida = llamar_llm(
            self.llm,
            prompts.SISTEMA_RESPUESTA.format(jurisdiccion=self.config.jurisdiccion),
            prompts.PLANTILLA_USUARIO.format(
                pregunta=pregunta,
                elementos=resumir_analisis(analisis),
                contexto=formatear_contexto(recuperacion.resultados),
            ),
        )
        advertencias: list[str] = []

        if not salida.strip():
            estado = "error_generacion"
            texto = ""
            advertencias.append(
                "El modelo generativo no produjo una respuesta visible."
            )
        else:
            advertencias: list[str] = []
            marca = RE_SUFICIENCIA.search(salida[:200])

            if marca:
                estado = ESTADOS[marca.group(1).upper()]
                texto = (salida[:marca.start()] + salida[marca.end():]).strip()
            else:
                estado = "parcial"
                texto = salida.strip()
                advertencias.append(
                    "El modelo no declaró la suficiencia. "
                    "La respuesta se clasificó provisionalmente como parcial."
                )

        # Etapa 5: verificación de citas.
        fuentes = [
            Fuente(i, r.fragmento.tipo, r.fragmento.titulo, r.fragmento.localizador(), r.fragmento.archivo,
                   r.fragmento.origen, r.fragmento.texto, r.score_denso)
            for i, r in enumerate(recuperacion.resultados, start=1)
        ]
        citadas, avisos_citas = verificar(
            texto, [f.texto for f in fuentes], pregunta,
            [r.fragmento.articulo for r in recuperacion.resultados if r.fragmento.articulo],
        )
        if estado not in {"insuficiente", "error_generacion"}:
            advertencias += avisos_citas
        for fuente in fuentes:
            fuente.citada = fuente.numero in citadas


        
        # Verificar que todas las citas correspondan a fuentes existentes
        ids_citados = {
            int(numero)
            for numero in re.findall(r"\[F(\d+)\]", texto, flags=re.IGNORECASE)
        }

        ids_validos = {fuente.numero for fuente in fuentes}

        ids_invalidos = ids_citados - ids_validos

        if ids_invalidos:
            advertencias.append(
                "La respuesta contiene citas a fuentes inexistentes: "
                + ", ".join(
                    f"[F{numero}]" for numero in sorted(ids_invalidos)
                )
            )


        return RespuestaRAG(
            pregunta, estado, texto, analisis, consultas, fuentes, advertencias,
            recuperacion.mejor_score, time.time() - inicio,
        )
