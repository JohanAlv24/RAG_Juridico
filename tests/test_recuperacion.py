from rag_juridico.bm25 import IndiceBM25, tokenizar
from rag_juridico.fragmentacion import Fragmento
from rag_juridico.indice_vectorial import IndiceMemoria
from rag_juridico.recuperacion import Resultado, fusion_rrf, seleccionar_con_cobertura


def test_tokenizar_quita_tildes_y_vacias():
    assert tokenizar("La Resolución del artículo 1546") == ["resolucion", "articulo", "1546"]


def test_bm25_prioriza_coincidencia_exacta():
    indice = IndiceBM25([
        "El deudor está en mora cuando no cumple en el término estipulado",
        "Artículo 1546 condición resolutoria tácita en contratos bilaterales",
        "La cláusula penal asegura el cumplimiento de la obligación",
    ])
    assert indice.buscar("articulo 1546", k=1)[0][0] == 1
    assert indice.buscar("clausula penal", k=1)[0][0] == 2
    assert indice.buscar("criptomonedas", k=3) == []


def test_rrf_premia_aparecer_en_varias_listas():
    puntajes = fusion_rrf([["a", "b", "c"], ["b", "a"], ["b"]], k=60)
    assert puntajes["b"] > puntajes["a"] > puntajes["c"]


def _resultado(tipo, puntaje):
    fragmento = Fragmento(f"{tipo}-{puntaje}", "d", tipo, "T", "a", "", 0, "texto")
    return Resultado(fragmento, puntaje)


def test_cobertura_garantiza_cada_tipo():
    ordenados = [_resultado("ley", 0.9), _resultado("ley", 0.8), _resultado("ley", 0.7),
                 _resultado("sentencia", 0.5), _resultado("caso", 0.4)]
    seleccion = seleccionar_con_cobertura(ordenados, k=3, min_por_tipo=1)
    assert [r.fragmento.tipo for r in seleccion] == ["ley", "sentencia", "caso"]
    assert len(seleccionar_con_cobertura(ordenados, k=3, min_por_tipo=0)) == 3
    assert [r.fragmento.tipo for r in seleccionar_con_cobertura(ordenados, 3, 0)] == ["ley"] * 3


def test_indice_memoria_coseno_y_persistencia(tmp_path):
    ruta = tmp_path / "v.npz"
    indice = IndiceMemoria(ruta)
    indice.recrear(2)
    indice.agregar(["x", "y"], [[1, 0], [0, 5]])
    assert indice.buscar([0, 1], k=1) == [("y", 1.0)]
    recargado = IndiceMemoria(ruta)
    assert recargado.contar() == 2 and recargado.buscar([2, 0], k=1)[0][0] == "x"
