from rag_juridico.carga import DocumentoFuente, cargar_corpus
from rag_juridico.fragmentacion import dividir_texto, fragmentar_documento, normalizar

from conftest import LEY, SENTENCIA


def _doc(tipo, texto, paginas=None, paginado=False):
    return DocumentoFuente("x/doc.txt", tipo, "Documento", "x/doc.txt", "", paginas or [texto], paginado)


def test_ley_un_fragmento_por_articulo():
    fragmentos = fragmentar_documento(_doc("ley", LEY))
    articulos = [f.articulo for f in fragmentos if f.articulo]
    assert articulos == ["1546", "1592", "1608", "1609", "1613"]
    f1546 = next(f for f in fragmentos if f.articulo == "1546")
    assert "condición resolutoria" in f1546.texto and "cláusula penal" not in f1546.texto
    assert "art. 1546" in f1546.localizador()
    assert f1546.texto_indexable().startswith("Ley | Documento | art. 1546")


def test_articulo_largo_se_subdivide_conservando_numero():
    largo = "ARTÍCULO 1. " + "Texto del artículo uno. " * 200
    texto = largo + "\n\nARTÍCULO 2. Corto.\n\nARTÍCULO 3. Otro corto pero con algo más de contenido útil."
    fragmentos = fragmentar_documento(_doc("ley", texto), tam=500, solape=50)
    del_uno = [f for f in fragmentos if f.articulo == "1"]
    assert len(del_uno) > 3
    assert all(len(f.texto) <= 500 for f in fragmentos)


def test_remision_en_minuscula_no_abre_articulo():
    texto = ("ARTÍCULO 10. Regla general sobre la mora del deudor en las obligaciones.\n"
             "conforme al\nartículo 99 de este código se aplicará lo dispuesto.\n\n"
             "ARTÍCULO 11. Otra regla relevante del régimen.\n\nARTÍCULO 12. Regla final del capítulo.")
    assert [f.articulo for f in fragmentar_documento(_doc("ley", texto))] == ["10", "11", "12"]


def test_sentencia_anota_seccion():
    fragmentos = fragmentar_documento(_doc("sentencia", SENTENCIA), tam=300, solape=40)
    secciones = {f.seccion for f in fragmentos}
    assert {"Antecedentes", "Consideraciones", "Resuelve"} <= secciones
    assert all(f.articulo is None for f in fragmentos)


def test_division_respeta_tamano_y_solapa():
    texto = " ".join(f"Oración número {i} del documento." for i in range(300))
    tramos = dividir_texto(texto, tam=400, solape=80)
    assert all(b - a <= 400 for a, b in tramos)
    assert tramos[0][0] == 0 and tramos[-1][1] == len(texto)
    assert all(tramos[i + 1][0] < tramos[i][1] for i in range(len(tramos) - 1))  # hay traslape


def test_paginas_se_asignan():
    paginas = ["Primera página. " * 30, "Segunda página con la cláusula penal. " * 30, "Tercera página. " * 30]
    fragmentos = fragmentar_documento(_doc("caso", "", paginas, paginado=True), tam=300, solape=0)
    assert {f.pagina for f in fragmentos} == {1, 2, 3}
    assert all("Segunda" in f.texto for f in fragmentos if f.pagina == 2 and "Primera" not in f.texto)
    assert "p. 2" in next(f for f in fragmentos if f.pagina == 2).localizador()


def test_normalizar_une_palabras_cortadas():
    assert normalizar("incumpli-\nmiento   del\r\ncontrato\n\n\n\nfin") == "incumplimiento del\ncontrato\n\nfin"


def test_ids_deterministas():
    a = fragmentar_documento(_doc("ley", LEY))
    b = fragmentar_documento(_doc("ley", LEY))
    assert [f.chunk_id for f in a] == [f.chunk_id for f in b]
    assert len({f.chunk_id for f in a}) == len(a)


def test_cargar_corpus(config):
    documentos, avisos = cargar_corpus(config.dir_datos, verbose=False)
    assert {d.tipo for d in documentos} == {"ley", "sentencia", "caso"}
    ley = next(d for d in documentos if d.tipo == "ley")
    assert ley.titulo == "Código Civil de prueba" and ley.origen.startswith("https://")
    assert sum("sin origen declarado" in a for a in avisos) == 2
