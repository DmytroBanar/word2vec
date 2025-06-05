import mysql.connector
from mysql.connector import Error

def get_db_connection():
    try:
        connection = mysql.connector.connect(
            host='localhost',
            database='Summaraize_application',
            user='root',
            password=''
        )
        if connection.is_connected():
            return connection
    except Error as e:
        print("Помилка підключення до бази даних:", e)
        return None