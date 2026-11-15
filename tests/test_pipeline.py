"""Pruebas de extremo a extremo con embeddings simulados, índice en memoria y LLM falso."""

import json

import pytest

from rag_juridico.evaluacion import comprobar, evaluar
from rag_juridico.indice_vectorial import IndiceMemoria
from rag_juridico.ingesta import CorpusVacio, ingestar, leer_fragmentos
from rag_juridico.pipeline import AsistenteJuridico, extraer_json
from rag_juridico.presentacion import a_markdown
from rag_juridico.simulado import EmbeddingsSimulados

from conftest import LLMFalso

CASO = ("Ana compró un inmueble a Luis y pagó todo el precio, pero Luis no lo entregó en la fecha "
        "pactada. ¿Qué pasa con el contrato de compraventa por ese incumplimiento?")


def _asistente(config, llm):
    indice = IndiceMemoria()
    ingestar(config, EmbeddingsSimulados(), indice, verbose=False)
    return AsistenteJuridico(config, EmbeddingsSimulados(), llm, indice, leer_fragmentos(config.ruta_fragmentos))


def test_ingesta_deja_manifiesto_y_fragmentos(config):
    manifiesto = ingestar(config, verbose=False)
    assert manifiesto["fragmentos"] == len(leer_fragmentos(config.ruta_fragmentos)) > 5
    assert set(manifiesto["fragmentos_por_tipo"]) == {"ley", "sentencia", "caso"}
    assert json.loads(config.ruta_manifiesto.read_text(encoding="utf-8"))["dimension"] == 384


def test_corpus_vacio(config, tmp_path):
    from dataclasses import replace
    with pytest.raises(CorpusVacio):
        ingestar(replace(config, dir_datos=tmp_path / "nada"), verbose=False)


def test_caso_respondido_con_fuentes_de_los_tres_tipos(config):
    llm = LLMFalso()
    respuesta = _asistente(config, llm).responder(CASO)
    assert respuesta.estado == "respondida"
    assert respuesta.respuesta.startswith("Procede la resolución")  # se quitó la línea SUFICIENCIA
    assert {f.tipo for f in respuesta.fuentes} == {"ley", "sentencia", "caso"}
    assert respuesta.fuentes[0].citada and not respuesta.fuentes[1].citada
    assert respuesta.analisis["tipo_contrato"] == "compraventa"
    assert len(respuesta.consultas) == 4  # pregunta + 2 consultas + pregunta jurídica
    assert len(llm.llamadas) == 2
    # el generador recibió el contexto numerado y los elementos del caso
    usuario = llm.llamadas[1][1][1]
    assert "[F1]" in usuario and "Tipo de contrato: compraventa" in usuario
    assert any(f.titulo == "Código Civil de prueba" and "art. 1546" in f.localizador for f in respuesta.fuentes)
    markdown = a_markdown(respuesta)
    assert "Fuentes recuperadas" in markdown and "https://ejemplo.test/cc" in markdown


def test_abstencion_sin_llamar_al_generador(config):
    llm = LLMFalso(analisis={"consultas_busqueda": ["receta paella mariscos"]})
    respuesta = _asistente(config, llm).responder("¿Cuál es la mejor receta de paella valenciana con mariscos?")
    assert respuesta.estado == "sin_contexto" and respuesta.fuentes == []
    assert "No tengo contexto suficiente" in respuesta.respuesta
    assert len(llm.llamadas) == 1  # solo el análisis; no se generó respuesta


def test_modelo_declara_insuficiente(config):
    llm = LLMFalso(respuesta="SUFICIENCIA: INSUFICIENTE\nEl contexto no trata contratos laborales.")
    respuesta = _asistente(config, llm).responder(CASO)
    assert respuesta.estado == "insuficiente" and respuesta.advertencias == []
    assert respuesta.fuentes  # las fuentes se muestran igual para poder revisar


def test_cita_inventada_genera_advertencia(config):
    llm = LLMFalso(respuesta="SUFICIENCIA: PARCIAL\nSegún el artículo 9999 procede [F1] y [F42].")
    respuesta = _asistente(config, llm).responder(CASO)
    assert respuesta.estado == "parcial"
    assert any("[F42]" in a for a in respuesta.advertencias)
    assert any("9999" in a for a in respuesta.advertencias)


def test_analisis_ilegible_usa_pregunta_original(config):
    llm = LLMFalso(analisis="lo siento, no puedo producir JSON")
    respuesta = _asistente(config, llm).responder(CASO)
    assert respuesta.analisis is None and respuesta.consultas == [CASO]
    assert respuesta.estado == "respondida"


def test_extraer_json_tolerante():
    assert extraer_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extraer_json("Claro: {\"a\": [1, 2]} fin") == {"a": [1, 2]}
    assert extraer_json("nada") is None and extraer_json("{roto") is None


def test_desde_storage_y_evaluacion(config, tmp_path):
    ingestar(config, verbose=False)
    asistente = AsistenteJuridico.desde_storage(config)  # modo simulado + índice en memoria persistido
    casos = tmp_path / "casos.json"
    casos.write_text(json.dumps({"casos": [{
        "id": "C1", "categoria": "habitual", "pregunta": CASO, "resultado_esperado": "resolución o cumplimiento",
        "estados_esperados": ["respondida", "parcial"], "tipos_fuente_esperados": ["ley"],
        "terminos_esperados": ["modo simulado", "inexistente"],
    }]}), encoding="utf-8")
    informe = evaluar(asistente, casos, tmp_path / "res", verbose=False)
    texto = informe.read_text(encoding="utf-8")
    assert "C1 — habitual" in texto and "Valoración manual" in texto and "inexistente" in texto
    chequeo = comprobar(json.loads(casos.read_text(encoding="utf-8"))["casos"][0], asistente.responder(CASO))
    assert chequeo == {"estado_ok": True, "tipos_faltantes": [], "terminos_faltantes": ["inexistente"]}


def test_indice_de_otro_modelo_se_rechaza(config):
    from dataclasses import replace
    ingestar(config, verbose=False)
    with pytest.raises(RuntimeError, match="Vuelve a ejecutar la ingesta"):
        AsistenteJuridico.desde_storage(replace(config, modo="nvidia"))
