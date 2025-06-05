from flask import Blueprint, request, jsonify, current_app
from db import get_db_connection
from werkzeug.security import check_password_hash
from flask_jwt_extended import (
    create_access_token,
    create_refresh_token,
    jwt_required,
    get_jwt_identity
)
from flask_mail import Mail, Message

login_bp = Blueprint('login_bp', __name__)

mail = Mail()  # mail ініціалізується в app і використовується тут

@login_bp.route('/login', methods=['POST'])
def login_user():
    conn = get_db_connection()
    if conn is None:
        return jsonify({"error": "Database connection failed"}), 500

    try:
        data = request.json
        email = data.get('email')
        password = data.get('password')

        if not email or not password:
            return jsonify({"error": "Будь ласка, введіть email та пароль"}), 400

        cursor = conn.cursor(dictionary=True)
        query = "SELECT id, email, password, is_verified FROM users WHERE email = %s"
        cursor.execute(query, (email,))
        user = cursor.fetchone()

        if not user or not check_password_hash(user['password'], password):
            return jsonify({"error": "Невірний email або пароль"}), 401

        if not user['is_verified']:
            return jsonify({"error": "Будь ласка, підтвердіть вашу електронну пошту перед входом"}), 400

        access_token = create_access_token(identity=user['id'])
        refresh_token = create_refresh_token(identity=user['id'])

        # Надсилаємо повідомлення на email про авторизацію
        try:
            msg = Message(
                subject="Успішна авторизація",
                sender=current_app.config['MAIL_USERNAME'],
                recipients=[user['email']],
                body="Ви щойно увійшли у свій акаунт. Якщо це були не ви — змініть пароль і зверніться до служби підтримки."
            )
            mail.send(msg)
        except Exception as mail_error:
            current_app.logger.warning(f"Не вдалося надіслати повідомлення про авторизацію: {mail_error}")

        return jsonify({
            "message": "Вхід успішний",
            "access_token": access_token,
            "refresh_token": refresh_token
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()
