from flask import Blueprint, request, jsonify
from db import get_db_connection

verify_bp = Blueprint('verify_bp', __name__)

@verify_bp.route('/verify-email', methods=['POST'])
def verify_email():
    conn = get_db_connection()
    if conn is None:
        return jsonify({"error": "Database connection failed"}), 500

    try:
        data = request.json
        email = data.get('email')
        code = data.get('code')

        cursor = conn.cursor()
        cursor.execute("SELECT verification_code FROM users WHERE email = %s", (email,))
        result = cursor.fetchone()

        if result is None:
            return jsonify({"error": "Користувача не знайдено"}), 404

        if result[0] == code:
            cursor.execute("UPDATE users SET is_verified = TRUE WHERE email = %s", (email,))
            conn.commit()
            return jsonify({"message": "Пошту успішно підтверджено"})
        else:
            return jsonify({"error": "Невірний код підтвердження"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()
