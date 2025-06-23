import mysql.connector
from mysql.connector import Error

DB_CONFIG = {
    'host': 'localhost',
    'database': 'Summaraize_application',
    'user': 'root',
    'password': ''
}

def get_db_connection():
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            return connection
    except Error as e:
        print("Помилка підключення до бази даних:", e)
    return None

def add_user_model(user_id, model_name, description, file_path):
    try:
        conn = get_db_connection()
        if not conn:
            print("DB ERROR: No connection")
            return
        c = conn.cursor()
        c.execute('''
            INSERT INTO user_models (user_id, model_name, description, file_path)
            VALUES (%s, %s, %s, %s)
        ''', (user_id, model_name, description, file_path))
        conn.commit()
        c.close()
        conn.close()
    except Exception as e:
        print("DB ERROR:", e)

def get_user_models(user_id):
    models = []
    try:
        conn = get_db_connection()
        if not conn:
            print("DB ERROR: No connection")
            return []
        c = conn.cursor()
        c.execute('''
            SELECT id, user_id, model_name, description, file_path, created_at
            FROM user_models
            WHERE user_id = %s OR user_id IS NULL
            ORDER BY created_at DESC
        ''', (user_id,))
        rows = c.fetchall()
        for row in rows:
            models.append({
                "id": row[0],
                "user_id": row[1],
                "model_name": row[2],
                "description": row[3],
                "file_path": row[4],
                "created_at": row[5]
            })
        c.close()
        conn.close()
    except Exception as e:
        print("DB ERROR:", e)
    return models

def update_user_model(model_id, user_id, data):
    try:
        conn = get_db_connection()
        if not conn:
            print("DB ERROR: No connection")
            return
        c = conn.cursor()
        c.execute('''
            UPDATE user_models
            SET model_name = %s, description = %s, file_path = %s
            WHERE id = %s AND user_id = %s
        ''', (data.get('model_name'), data.get('description'), data.get('file_path'), model_id, user_id))
        conn.commit()
        c.close()
        conn.close()
    except Exception as e:
        print("DB ERROR:", e)

def delete_user_model(model_id, user_id):
    try:
        conn = get_db_connection()
        if not conn:
            print("DB ERROR: No connection")
            return
        c = conn.cursor()
        c.execute('''
            DELETE FROM user_models
            WHERE id = %s AND user_id = %s
        ''', (model_id, user_id))
        conn.commit()
        c.close()
        conn.close()
    except Exception as e:
        print("DB ERROR:", e)