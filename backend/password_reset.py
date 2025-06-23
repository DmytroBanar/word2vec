from flask import Blueprint, request, jsonify, current_app, render_template
from db import get_db_connection
from flask_mail import Mail, Message
from werkzeug.security import generate_password_hash, check_password_hash
import random
import string
from datetime import datetime, timedelta

password_reset_bp = Blueprint('password_reset_bp', __name__)
mail = Mail()  # mail ініціалізується в app і використовується тут

# Генерація випадкового токену для скидання паролю
def generate_reset_token(length=32):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))

@password_reset_bp.route('/forgot-password', methods=['POST'])
def forgot_password():
    conn = get_db_connection()
    if conn is None:
        return jsonify({"error": "Database connection failed"}), 500

    try:
        data = request.json
        email = data.get('email')

        cursor = conn.cursor(dictionary=True)


        # Перевірка, чи існує користувач з таким email
        cursor.execute("SELECT id, email FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()
        if not user:
            return jsonify({"error": "Користувача з таким email не знайдено"}), 404

        # Генерація токену для скидання паролю
        reset_token = generate_reset_token()

        # Запис токену та терміну дії в базу даних (наприклад, термін дії — 1 година)
        expiration_time = datetime.utcnow() + timedelta(minutes=15)
        cursor.execute("""
            UPDATE users
            SET reset_token = %s, reset_token_expiration = %s
            WHERE email = %s
        """, (reset_token, expiration_time, email))
        conn.commit()

        # Надсилання email з посиланням для скидання паролю
        reset_url = f"{current_app.config['CLIENT_URL']}/reset-password/{reset_token}"
        msg = Message(
            subject="Відновлення паролю",
            sender=current_app.config['MAIL_USERNAME'],
            recipients=[email],
            body=f"Для скидання паролю перейдіть за посиланням: {reset_url}"
        )
        mail.send(msg)

        return jsonify({"message": "Лист з інструкціями надіслано на вашу пошту."}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()


@password_reset_bp.route('/reset-password/<reset_token>', methods=['GET'])
def reset_password_form(reset_token):
    conn = get_db_connection()
    if conn is None:
        return jsonify({"error": "Database connection failed"}), 500

    try:
        # Перевіряємо, чи існує користувач із таким токеном
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT id, reset_token, reset_token_expiration
            FROM users
            WHERE reset_token = %s
        """, (reset_token,))
        user = cursor.fetchone()

        if not user or user['reset_token_expiration'] < datetime.utcnow():
            return jsonify({"error": "Токен не знайдено або термін його дії вичерпано"}), 400

        # Якщо все ок, рендеримо шаблон для введення нового паролю
        return render_template('components/ResetPasswordForm.jsx', reset_token=reset_token)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@password_reset_bp.route('/verify-reset-token/<reset_token>', methods=['GET'])
def verify_reset_token(reset_token):
    conn = get_db_connection()
    if conn is None:
        return jsonify({"error": "Database connection failed"}), 500

    try:
        cursor = conn.cursor(dictionary=True)  # Додаємо цей параметр
        cursor.execute("""
            SELECT id, reset_token, reset_token_expiration
            FROM users
            WHERE reset_token = %s
        """, (reset_token,))
        user = cursor.fetchone()

        if not user or not user['reset_token_expiration'] or user['reset_token_expiration'] < datetime.utcnow():
            return jsonify({"error": "Токен не знайдено або термін його дії вичерпано"}), 400

        return jsonify({"message": "Токен дійсний"}), 200

    except Exception as e:
        current_app.logger.error(f"Error during reset token verification: {str(e)}")
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@password_reset_bp.route('/reset-password', methods=['POST'])
def reset_password():
    conn = get_db_connection()
    if conn is None:
        return jsonify({"error": "Database connection failed"}), 500

    try:
        data = request.get_json()
        token = data.get('token')
        new_password = data.get('password')

        if not token or not new_password:
            return jsonify({"error": "Неповні дані"}), 400

        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT id, email, reset_token_expiration FROM users
            WHERE reset_token = %s
        """, (token,))
        user = cursor.fetchone()

        if not user or user['reset_token_expiration'] < datetime.utcnow():
            return jsonify({"error": "Невалідний або прострочений токен"}), 400

        # ⏺ Зберігаємо email ПЕРЕД обнуленням токена
        user_email = user['email']

        hashed_password = generate_password_hash(new_password)
        cursor.execute("""
            UPDATE users
            SET password = %s, reset_token = NULL, reset_token_expiration = NULL
            WHERE reset_token = %s
        """, (hashed_password, token))
        conn.commit()

        # Надсилання повідомлення
        msg = Message(
            subject="Ваш пароль змінено",
            sender=current_app.config['MAIL_USERNAME'],
            recipients=[user_email],
            body="Ваш пароль було успішно змінено. Якщо ви цього не робили — негайно змініть пароль або зверніться до служби підтримки."
        )
        mail.send(msg)

        return jsonify({"message": "Пароль успішно змінено."}), 200

    except Exception as e:
        current_app.logger.error(f"Password reset error: {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        conn.close()