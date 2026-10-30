"""Actor de Apify: clasifica conexiones de red (NSL-KDD) con un Random Forest.

Flujo pensado para Make:
    Google Sheets (filas) -> JSON `conexiones` -> este Actor -> dataset de resultados -> Google Sheets

Entrada : ver .actor/input_schema.json
Salida  : un item por conexión en el dataset por defecto, con columnas planas
          (id, estado, es_ataque, prob_ataque, categoria, error) listas para pegar en una Sheet.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from apify import Actor

ARCHIVO_MODELO = 'modelo_full.pkl'
RUTA_MODELO = Path(__file__).parent

# Columnas de texto. El resto de columnas de entrada del modelo son numéricas.
COLUMNAS_CATEGORICAS = {'protocol_type', 'flag', 'service'}


def a_numero(valor, nombre: str) -> float:
    """Convierte lo que llega de la Sheet/Make a número.

    Acepta números, texto numérico y coma decimal ("0,25"), que es lo que suele
    devolver una Sheet configurada en español. Rechaza vacíos y texto no numérico.
    """
    if isinstance(valor, bool):
        return float(valor)
    if isinstance(valor, (int, float)):
        return float(valor)
    if isinstance(valor, str):
        texto = valor.strip()
        if not texto:
            raise ValueError(f'"{nombre}" está vacío')
        if ',' in texto and '.' in texto:      # "1.234,5" -> 1234.5
            texto = texto.replace('.', '').replace(',', '.')
        else:                                    # "0,25" -> 0.25
            texto = texto.replace(',', '.')
        try:
            return float(texto)
        except ValueError:
            raise ValueError(f'"{nombre}" no es un número: {valor!r}') from None
    raise ValueError(f'"{nombre}" falta o no es un número')


def a_texto(valor, nombre: str) -> str:
    if valor is None or not str(valor).strip():
        raise ValueError(f'"{nombre}" está vacío o falta')
    return str(valor).strip()


def normalizar_conexiones(crudo) -> list[dict]:
    """Make a veces manda el array como texto JSON o un único objeto; se aceptan los tres casos."""
    if isinstance(crudo, str):
        crudo = json.loads(crudo)
    if isinstance(crudo, dict):
        crudo = [crudo]
    if not isinstance(crudo, list) or not crudo:
        raise ValueError('"conexiones" debe ser una lista con al menos una conexión')
    return crudo


def preparar_fila(fila: dict, columnas_entrada: list[str]) -> dict:
    """Valida y convierte una fila. Lanza ValueError con un mensaje claro si algo está mal."""
    if not isinstance(fila, dict):
        raise ValueError('la fila no es un objeto JSON')
    faltan = [c for c in columnas_entrada if c not in fila]
    if faltan:
        raise ValueError(f'faltan columnas: {", ".join(faltan)}')
    limpia = {}
    for col in columnas_entrada:
        if col in COLUMNAS_CATEGORICAS:
            limpia[col] = a_texto(fila[col], col)
        else:
            limpia[col] = a_numero(fila[col], col)
    return limpia


def codificar(df_entrada: pd.DataFrame, columnas_modelo: list[str]) -> pd.DataFrame:
    """One-hot igual que en el entrenamiento: las columnas que faltan quedan en 0 y las
    desconocidas (por ejemplo un servicio nuevo) se descartan sin romper nada."""
    X = pd.get_dummies(df_entrada)
    return X.reindex(columns=columnas_modelo, fill_value=0)


async def main() -> None:
    async with Actor:
        entrada = await Actor.get_input() or {}

        try:
            conexiones = normalizar_conexiones(entrada.get('conexiones'))
        except (ValueError, json.JSONDecodeError) as e:
            await Actor.fail(status_message=f'Entrada inválida: {e}')
            return

        umbral = float(entrada.get('umbralAtaque', 0.5))
        if not 0 <= umbral <= 1:
            await Actor.fail(status_message='"umbralAtaque" debe estar entre 0 y 1')
            return

        paquete = joblib.load(RUTA_MODELO / ARCHIVO_MODELO)
        if paquete['sklearn_version'] != sklearn.__version__:
            Actor.log.warning(
                f'El modelo se entrenó con scikit-learn {paquete["sklearn_version"]} '
                f'y el Actor usa {sklearn.__version__}. Fijá la misma versión en requirements.txt.'
            )
        Actor.log.info(f'Modelo cargado ({paquete["modo_entrenamiento"]}). Conexiones recibidas: {len(conexiones)}')

        columnas_entrada = paquete['columnas_entrada']
        resultados: list[dict | None] = [None] * len(conexiones)
        validas: list[dict] = []
        posiciones: list[int] = []

        # 1) Validar cada fila; una fila mala no frena al resto.
        for i, fila in enumerate(conexiones):
            id_fila = fila.get('id', i + 1) if isinstance(fila, dict) else i + 1
            try:
                validas.append(preparar_fila(fila, columnas_entrada))
                posiciones.append(i)
            except ValueError as e:
                resultados[i] = {
                    'id': id_fila, 'estado': 'error', 'es_ataque': None,
                    'prob_ataque': None, 'categoria': None, 'error': str(e),
                }

        # 2) Predecir todas las filas válidas juntas.
        if validas:
            X = codificar(pd.DataFrame(validas), paquete['columnas'])
            prob = paquete['modelo_binario'].predict_proba(X)[:, 1]
            categoria = paquete['modelo_multiclase'].predict(X)
            for k, i in enumerate(posiciones):
                fila = conexiones[i]
                resultados[i] = {
                    'id': fila.get('id', i + 1),
                    'estado': 'ok',
                    'es_ataque': bool(prob[k] >= umbral),
                    'prob_ataque': round(float(prob[k]), 4),
                    'categoria': str(categoria[k]),
                    'error': None,
                }

        # 3) Guardar en el dataset (una fila de salida por cada fila de entrada).
        await Actor.push_data(resultados)

        n_ok = len(validas)
        n_ataques = sum(1 for r in resultados if r['estado'] == 'ok' and r['es_ataque'])
        mensaje = f'{n_ok}/{len(conexiones)} filas procesadas, {n_ataques} marcadas como ataque'
        Actor.log.info(mensaje)
        await Actor.set_status_message(mensaje)