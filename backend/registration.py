from flask import Blueprint, request, jsonify, current_app
from db import get_db_connection
from flask_mail import Mail, Message
from werkzeug.security import generate_password_hash
import random
import string

registration_bp = Blueprint('registration_bp', __name__)
mail = Mail()  # mail ініціалізується в app і використовується тут

# Генерація випадкового коду підтвердження
def generate_verification_code(length=6):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))

@registration_bp.route('/register', methods=['POST'])
def register_user():
    conn = get_db_connection()
    if conn is None:
        return jsonify({"error": "Database connection failed"}), 500

    try:
        data = request.json
        email = data.get('email')
        password = data.get('password')
        confirm_password = data.get('confirmPassword')

        # Перевірка, чи збігаються паролі
        if password != confirm_password:
            return jsonify({"error": "Паролі не співпадають"}), 400

        cursor = conn.cursor()

        # Перевірка, чи вже існує користувач з таким email
        cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
        if cursor.fetchone():
            return jsonify({"error": "Користувач з таким email вже існує"}), 400

        # Хешування пароля
        hashed_password = generate_password_hash(password)

        # Генерація коду підтвердження
        verification_code = generate_verification_code()

        query = """
            INSERT INTO users (email, password, verification_code, is_verified)
            VALUES (%s, %s, %s, %s)
        """
        cursor.execute(query, (email, hashed_password, verification_code, False))
        conn.commit()

        # Надсилання email з кодом
        msg = Message(
            subject="Підтвердження електронної пошти",
            sender=current_app.config['MAIL_USERNAME'],
            recipients=[email],
            body=f"Дякуємо за реєстрацію! Ваш код підтвердження: {verification_code}"
        )
        mail.send(msg)

        return jsonify({"message": "Користувача зареєстровано. Перевірте email для підтвердження."}), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()
