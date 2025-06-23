from flask import Flask, request, jsonify
from gensim.models import KeyedVectors
from flask_cors import CORS
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import base64
import os
from werkzeug.utils import secure_filename
from utils.preprocessing import preprocess_text
from io import BytesIO
import numpy as np
from matplotlib.lines import Line2D
import networkx as nx
from registration import registration_bp, mail
from verify_email import verify_bp
from auth import login_bp
from password_reset import password_reset_bp
from flask_jwt_extended import JWTManager, jwt_required, get_jwt_identity, verify_jwt_in_request
from datetime import timedelta
from dotenv import load_dotenv
from utils.preprocessing import preprocess_text, extract_keywords
import traceback
from collections import Counter
from db import add_user_model, get_user_models, update_user_model, delete_user_model
import sqlite3

load_dotenv()

app = Flask(__name__)
CORS(app, supports_credentials=True)

app.config['CLIENT_URL'] = os.getenv("CLIENT_URL", "http://localhost:9000")

app.config.update({
    "MAIL_SERVER": "smtp.gmail.com",
    "MAIL_PORT": 587,
    "MAIL_USERNAME": "banar.dima04@gmail.com",
    "MAIL_PASSWORD": "wyfd achx hxew ieux",
    "MAIL_USE_TLS": True,
    "MAIL_USE_SSL": False,
    "JWT_SECRET_KEY": os.getenv("JWT_SECRET_KEY"),
    "JWT_ACCESS_TOKEN_EXPIRES": timedelta(days=15),
    "JWT_REFRESH_TOKEN_EXPIRES": timedelta(days=30),
})

jwt = JWTManager(app)
mail.init_app(app)

app.register_blueprint(registration_bp)
app.register_blueprint(verify_bp)
app.register_blueprint(login_bp)
app.register_blueprint(password_reset_bp)

CORS(app)

UPLOAD_FOLDER = 'upload'
MODEL_FOLDER = 'final_model'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

def load_model(model_name=None):
    try:
        if model_name is None:
            return KeyedVectors.load(os.path.join("first_model", "Lucifer_main.kv"))
        else:
            return KeyedVectors.load(os.path.join(MODEL_FOLDER, model_name))
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        filename = secure_filename(file.filename)  # напр. harrypotter.txt
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        base_name = os.path.splitext(filename)[0]  # буде 'harrypotter'
        try:
            model_path = preprocess_text(file_path, MODEL_FOLDER, base_name)
            kv_filename = os.path.basename(model_path)  # буде 'harrypotter.kv'
            return jsonify({"message": "File uploaded and processed successfully", "file_path": kv_filename, "active_model": kv_filename})
        except Exception as e:
            return jsonify({"error": f"File uploaded but processing failed: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/files', methods=['GET'])
def list_files():
    try:
        files = os.listdir(MODEL_FOLDER)
        return jsonify(files)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/similarity', methods=['GET'])
def similarity():
    model_path = request.args.get('filename', None)
    model = load_model(model_path)
    if model is None:
        return jsonify({"error": "Model is not loaded"}), 500

    try:
        word = request.args.get('word')
        num_neighbors = int(request.args.get('num_neighbors', 10))
        similar_words = model.most_similar(word, topn=num_neighbors)
        return jsonify(similar_words)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/analogy', methods=['POST'])
@jwt_required()
def analogy():
    model_path = request.args.get('filename', None)
    model = load_model(model_path)
    if model is None:
        return jsonify({"error": "Model is not loaded"}), 500

    try:
        data = request.json
        word = data.get('word')
        pair = data.get('pair', [])
        num_analogies = int(data.get('num_analogies', 10))

        # Перевірка вхідних даних
        if not word or not isinstance(pair, list) or len(pair) != 2:
            return jsonify({"error": "Provide 'word' and a pair of two elements."}), 400

        all_words = [word, pair[0], pair[1]]
        not_found = [w for w in all_words if w not in model]
        if not_found:
            return jsonify({"error": f"Word(s) not in vocabulary: {', '.join(not_found)}"}), 400

        # Аналогія: word is to pair[0] as ? is to pair[1]
        analogy_words = model.most_similar(
            positive=[word, pair[1]],
            negative=[pair[0]],
            topn=num_analogies + len(all_words)
        )

        # Фільтруємо вхідні слова з результату
        filtered = [w for w in analogy_words if w[0] not in all_words][:num_analogies]

        return jsonify(filtered)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/plot_similar_words_clusters', methods=['GET'])
@jwt_required()
def plot_similar_words_clusters():
    model_path = request.args.get('filename', None)
    model = load_model(model_path)
    if model is None:
        return jsonify({"error": "Model is not loaded"}), 500

    try:
        word = request.args.get('word')
        num_neighbors = int(request.args.get('num_neighbors', 10))
        num_clusters = int(request.args.get('num_clusters', 5))

        if not word:
            return jsonify({"error": "Please provide a 'word' parameter."}), 400

        similar_words = model.most_similar(word, topn=num_neighbors)
        words = [word] + [w[0] for w in similar_words]  # Додаємо центральне слово
        word_vectors = [model[w] for w in words if w in model]

        if not word_vectors:
            return jsonify({"error": "No valid words found in the model."}), 400

        kmeans = KMeans(n_clusters=num_clusters)
        clusters = kmeans.fit_predict(word_vectors)

        pca = PCA(n_components=2)
        word_vectors_pca = pca.fit_transform(word_vectors)

        # Додаємо індекс центрального слова (0)
        return jsonify({
            "points": [{"x": float(x), "y": float(y)} for x, y in word_vectors_pca],
            "labels": words,
            "clusters": [int(c) for c in clusters],
            "center_index": 0
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/plot_character_connections', methods=['GET'])
@jwt_required()
def plot_character_connections():
    model_path = request.args.get('filename', None)
    model = load_model(model_path)
    if model is None:
        return jsonify({"error": "Model is not loaded"}), 500
    try:
        characters = request.args.get('characters')
        if not characters:
            return jsonify({"error": "Please provide 'characters' parameter."}), 400
        characters = [char.strip() for char in characters.split(',')]
        valid_characters = [char for char in characters if char in model]
        if not valid_characters:
            return jsonify({"error": "No valid characters found in the model."}), 400
        G = nx.Graph()
        for char in valid_characters:
            G.add_node(char)
        for i, char1 in enumerate(valid_characters):
            for char2 in valid_characters[i+1:]:
                similarity = model.similarity(char1, char2)
                print(f"{char1} - {char2}: {similarity}")  # Діагностика
                if 0.6 < similarity:
                    G.add_edge(char1, char2, weight=similarity)
        if G.number_of_edges() == 0:
            all_pairs = []
            for i, char1 in enumerate(valid_characters):
                for char2 in valid_characters[i+1:]:
                    similarity = model.similarity(char1, char2)
                    all_pairs.append((char1, char2, similarity))
            all_pairs.sort(key=lambda x: -x[2])
            for char1, char2, similarity in all_pairs[:3]:
                G.add_edge(char1, char2, weight=similarity)
        pos = nx.spring_layout(G, seed=42)
        nodes = [{"id": n, "x": float(pos[n][0]), "y": float(pos[n][1])} for n in G.nodes()]
        edges = [
            {
                "source": u,
                "target": v,
                "weight": float(G[u][v]["weight"])
            }
            for u, v in G.edges()
        ]

        return jsonify({"nodes": nodes, "edges": edges})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/keywords', methods=['GET'])
def get_keywords():
    try:
        model_name = request.args.get('filename', 'Lucifer_main.kv')
        topn = int(request.args.get('topn', 10))
        model_path = os.path.join(MODEL_FOLDER, model_name)
        if not os.path.exists(model_path):
            return jsonify({"error": "Model not found"}), 404
        keywords = extract_keywords(model_path, topn=topn)
        return jsonify({"keywords": keywords})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/model_stats', methods=['GET'])
def model_stats():
    model_name = request.args.get('filename')
    if not model_name:
        return jsonify({"error": "No model specified"}), 400
    model_path = os.path.join(MODEL_FOLDER, model_name)
    if not os.path.exists(model_path):
        return jsonify({"error": "Model not found"}), 404
    model = KeyedVectors.load(model_path)
    vocab = model.index_to_key
    word_lengths = [len(w) for w in vocab]
    length_counts = dict(Counter(word_lengths))
    length_counts = {int(k): int(v) for k, v in length_counts.items()}
    top_words = vocab[:10]
    top_word_freqs = []
    if hasattr(model, 'get_vecattr'):
        for w in top_words:
            try:
                top_word_freqs.append(int(model.get_vecattr(w, "count")))
            except Exception:
                top_word_freqs.append(1)
    elif hasattr(model, 'vocab'):
        for w in top_words:
            try:
                top_word_freqs.append(int(model.vocab[w].count))
            except Exception:
                top_word_freqs.append(1)
    else:
        top_word_freqs = [1 for _ in top_words]
    vectors = np.array([model[w] for w in top_words])
    pca = PCA(n_components=2)
    coords = pca.fit_transform(vectors)
    pca_points = [{"word": w, "x": float(x), "y": float(y)} for w, (x, y) in zip(top_words, coords)]
    return jsonify({
        "vocab_size": int(len(vocab)),
        "vector_size": int(model.vector_size),
        "length_counts": length_counts,
        "top_words": list(top_words),
        "top_word_freqs": list(top_word_freqs),
        "pca_points": pca_points
    })

@app.route('/user_models', methods=['GET'])
def list_user_models():
    verify_jwt_in_request(optional=True)
    user_id = get_jwt_identity()
    if user_id is not None:
        models = get_user_models(user_id)
    else:
        models = get_user_models(None)
    return jsonify(models)

@app.route('/user_models', methods=['POST'])
@jwt_required()
def upload_user_model():
    user_id = get_jwt_identity()
    print("user_id in upload_user_model:", user_id)
    data = request.json
    add_user_model(
        user_id=user_id,
        model_name=data['model_name'],
        description=data.get('description', ''),
        file_path=data['file_path']
    )
    return jsonify({"status": "ok"})

@app.route('/user_models/<int:model_id>', methods=['PUT'])
@jwt_required()
def edit_user_model(model_id):
    user_id = get_jwt_identity()
    data = request.json
    update_user_model(model_id, user_id, data)
    return jsonify({"status": "updated"})

@app.route('/user_models/<int:model_id>', methods=['DELETE'])
@jwt_required()
def remove_user_model(model_id):
    user_id = get_jwt_identity()
    delete_user_model(model_id, user_id)
    return jsonify({"status": "deleted"})

if __name__ == "__main__":
    app.run(debug=True)