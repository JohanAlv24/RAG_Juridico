from rag_juridico.citas import extraer_citas, verificar

FUENTES = ["ARTÍCULO 1546. Condición resolutoria tácita.", "Sentencia SC0001-2020 sobre resolución."]


def test_extraer_citas_varios_formatos():
    assert extraer_citas("Uno [F1]. Dos [F2][F1]. Tres [F3, F4] y [F 5].") == [1, 2, 3, 4, 5]
    assert extraer_citas("Sin citas [nota] F1") == []


def test_respuesta_respaldada_no_genera_advertencias():
    validas, avisos = verificar("Según el artículo 1546 procede resolver [F1]; ver SC0001-2020 [F2].", FUENTES)
    assert validas == [1, 2] and avisos == []


def test_cita_inexistente():
    validas, avisos = verificar("Procede [F1] y también [F7].", FUENTES)
    assert validas == [1] and any("[F7]" in a for a in avisos)


def test_sin_citas():
    _, avisos = verificar("Procede la resolución.", FUENTES)
    assert any("no cita" in a for a in avisos)


def test_articulo_y_sentencia_no_respaldados():
    _, avisos = verificar("Los artículos 1546 y 1602, y la sentencia C-1008 de 2010, lo confirman [F1].", FUENTES)
    assert any("1602" in a and "1546" not in a for a in avisos)
    assert any("C-1008" in a for a in avisos)


def test_articulo_citado_por_el_usuario_no_alarma():
    _, avisos = verificar("La cláusula del artículo 5 del contrato aplica [F1].", FUENTES,
                          pregunta="El artículo 5 del contrato fija una multa.")
    assert avisos == []
