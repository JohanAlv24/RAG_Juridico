"""Utilidades de prueba: corpus sintético y modelos falsos (no se usa red ni API).

Los textos del corpus de prueba son resúmenes inventados para las pruebas;
NO son el texto oficial de ninguna norma ni sentencia.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_juridico.config import Config  # noqa: E402

LEY = """CÓDIGO CIVIL DE PRUEBA (texto sintético)

ARTÍCULO 1546. En los contratos bilaterales va envuelta la condición resolutoria en caso de no
cumplirse por uno de los contratantes lo pactado. En tal caso el otro contratante podrá pedir a su
arbitrio la resolución o el cumplimiento del contrato, con indemnización de perjuicios.

ARTÍCULO 1592. La cláusula penal es aquella en que una persona, para asegurar el cumplimiento de
una obligación, se sujeta a una pena que consiste en dar o hacer algo en caso de no ejecutar o
retardar la obligación principal.

ARTÍCULO 1608. El deudor está en mora cuando no ha cumplido la obligación dentro del término
estipulado, salvo que la ley exija requerimiento para constituirlo en mora.

ARTÍCULO 1609. En los contratos bilaterales ninguno de los contratantes está en mora dejando de
cumplir lo pactado, mientras el otro no lo cumpla por su parte o no se allane a cumplirlo.

ARTÍCULO 1613. La indemnización de perjuicios comprende el daño emergente y el lucro cesante,
ya provenga de no haberse cumplido la obligación o de haberse retardado el cumplimiento.
"""

SENTENCIA = """SENTENCIA SC0001-2020 (texto sintético)

ANTECEDENTES

El comprador demandó la resolución del contrato de compraventa porque el vendedor no entregó el
inmueble en la fecha pactada, pese a que el precio fue pagado en su totalidad.

CONSIDERACIONES

La acción resolutoria exige que el demandante haya cumplido o se haya allanado a cumplir sus
obligaciones. El contratante cumplido puede optar entre la resolución y el cumplimiento, en ambos
casos con indemnización de perjuicios.

RESUELVE

Declarar resuelto el contrato de compraventa y condenar al vendedor a restituir el precio y a pagar
los perjuicios causados.
"""

CASO = """Caso de estudio (sintético): arrendamiento de local comercial. El arrendatario dejó de pagar
tres cánones consecutivos. El contrato incluía una cláusula penal equivalente a dos cánones. El
arrendador pidió la terminación del contrato, la restitución del inmueble y el pago de la pena.
"""

ANALISIS = {
    "tipo_contrato": "compraventa",
    "partes": [{"nombre": "Ana", "rol": "compradora"}, {"nombre": "Luis", "rol": "vendedor"}],
    "obligaciones": ["entregar el inmueble", "pagar el precio"],
    "incumplimiento": "el vendedor no entregó el inmueble",
    "causa": "imputable",
    "clausulas_relevantes": [],
    "pregunta_juridica": "¿Puede la compradora pedir la resolución del contrato con perjuicios?",
    "datos_faltantes": [],
    "consultas_busqueda": ["condición resolutoria contratos bilaterales", "indemnización de perjuicios mora"],
}


class Mensaje:
    def __init__(self, content):
        self.content = content


class LLMFalso:
    """Devuelve respuestas programadas y registra cada llamada."""

    def __init__(self, analisis=None, respuesta="SUFICIENCIA: SUFICIENTE\nProcede la resolución con perjuicios [F1]."):
        self.analisis = json.dumps(ANALISIS if analisis is None else analisis, ensure_ascii=False) \
            if not isinstance(analisis, str) else analisis
        self.respuesta = respuesta
        self.llamadas = []

    def invoke(self, mensajes):
        self.llamadas.append(mensajes)
        return Mensaje(self.analisis if "JSON" in mensajes[0][1] else self.respuesta)


@pytest.fixture
def config(tmp_path) -> Config:
    datos = tmp_path / "data"
    for carpeta, nombre, texto in [
        ("leyes", "codigo_civil_prueba.txt", LEY),
        ("sentencias", "sc0001_2020.txt", SENTENCIA),
        ("casos", "arrendamiento.txt", CASO),
    ]:
        (datos / carpeta).mkdir(parents=True)
        (datos / carpeta / nombre).write_text(texto, encoding="utf-8")
    (datos / "fuentes.json").write_text(json.dumps({
        "_nota": "claves que empiezan por _ se ignoran",
        "leyes/codigo_civil_prueba.txt": {"titulo": "Código Civil de prueba", "origen": "https://ejemplo.test/cc"},
    }), encoding="utf-8")
    return Config(
        dir_datos=datos, dir_storage=tmp_path / "storage", modo="simulado", backend_vectorial="memoria",
        umbral_minimo=0.05, umbral_suficiencia=0.2, k_final=4,
    )
