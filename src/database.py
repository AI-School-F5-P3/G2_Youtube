import os
import sqlite3


def create_database(db_name="db/predicciones.db"):
    '''
    Create a database to store the predictions.

    Parameters:
        db_name (str): The name of the database.

    Returns:
        sqlite3.Connection: A connection to the database.
    '''
    os.makedirs(os.path.dirname(db_name), exist_ok=True)
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predicciones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            comentario TEXT,
            etiqueta TEXT,
            confianza REAL
        )
    ''')
    conn.commit()
    conn.close()


def save_prediction_to_db(comments, predictions, db_name="db/predicciones.db"):
    '''
    Guarda las predicciones en la base de datos.

    Parameters:
        comments (list): Lista de comentarios.
        predictions (list): Lista de tuplas (etiqueta, confianza).
        db_name (str): Nombre de la base de datos.
    '''
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        for comment, (label, confidence) in zip(comments, predictions):
            cursor.execute('''
                INSERT INTO predicciones (comentario, etiqueta, confianza)
                VALUES (?, ?, ?)
            ''', (comment, label, confidence))

        conn.commit()
    except Exception as e:
        print(f"Error al guardar en la base de datos: {e}")
    finally:
        conn.close()


def get_predictions(db_name="db/predicciones.db", limit=100):
    '''
    Obtiene las predicciones almacenadas en la base de datos.

    Parameters:
        db_name (str): Nombre de la base de datos.
        limit (int): Número máximo de registros a retornar.

    Returns:
        list: Lista de tuplas con las predicciones.
    '''
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT id, comentario, etiqueta, confianza
            FROM predicciones
            ORDER BY id DESC
            LIMIT ?
        ''', (limit,))

        predictions = cursor.fetchall()
        return predictions
    except Exception as e:
        print(f"Error al obtener predicciones: {e}")
        return []
    finally:
        conn.close()
