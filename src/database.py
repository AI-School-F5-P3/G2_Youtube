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
    Save the predictions to the database.

    Parameters:
        comments (list): A list of comments.
        predictions (list): A list of predictions.
        db_name (str): The name of the database.
    '''
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    for comment, (label, confidence) in zip(comments, predictions):
        cursor.execute('''
            INSERT INTO predicciones (comentario, etiqueta, confianza)
            VALUES (?, ?, ?)
        ''', (comment, label, confidence))
    conn.commit()
    conn.close()
