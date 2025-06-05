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
from flask_jwt_extended import JWTManager, jwt_required
from datetime import timedelta
from dotenv import load_dotenv
from utils.preprocessing import preprocess_text, extract_keywords
import traceback

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
    "JWT_ACCESS_TOKEN_EXPIRES": timedelta(minutes=15),
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

        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        try:
            model_path = preprocess_text(file_path, MODEL_FOLDER, filename)
            return jsonify({"message": "File uploaded and processed successfully", "file_path": file_path, "active_model": filename})
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
        print("Received JSON data:", data)

        word = data.get('word')
        pair = data.get('pair', [])
        num_analogies = data.get('num_analogies', 10)

        if not word or len(pair) != 2:
            return jsonify({"error": "Provide 'word' and a pair of two elements."}), 400

        analogy_words = model.most_similar(positive=[word, pair[1]], negative=[pair[0]], topn=num_analogies)
        return jsonify(analogy_words)

    except Exception as e:
        print("Error in analogy():", str(e))
        traceback.print_exc()
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
        words = [w[0] for w in similar_words]
        word_vectors = [model[w] for w in words if w in model]

        if not word_vectors:
            return jsonify({"error": "No valid words found in the model."}), 400

        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(word_vectors)
        labels = kmeans.labels_

        pca = PCA(n_components=2)
        word_vectors_pca = pca.fit_transform(word_vectors)

        plt.figure(figsize=(10, 10))
        cluster_colors = plt.cm.viridis(np.linspace(0, 1, num_clusters))

        for i, word in enumerate(words):
            plt.scatter(word_vectors_pca[i, 0], word_vectors_pca[i, 1], color=cluster_colors[labels[i]], alpha=0.7)
            plt.annotate(word, (word_vectors_pca[i, 0], word_vectors_pca[i, 1]))

        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)

        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return jsonify({"image": img_base64})
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
                if similarity > 0.5:
                    G.add_edge(char1, char2, weight=similarity)

        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(G)
        nx.draw_networkx_nodes(G, pos, node_size=700)
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
        nx.draw_networkx_edges(G, pos, width=2, alpha=0.5, edge_color='b')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'))

        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)

        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return jsonify({"image": img_base64})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/keywords', methods=['GET'])
def get_keywords():
    try:
        # Користувач може вручну задати кількість слів через параметр topn
        model_name = request.args.get('filename', 'Lucifer_main.kv')
        topn = int(request.args.get('topn', 10))  # topn - кількість ключових слів
        model_path = os.path.join(MODEL_FOLDER, model_name)
        if not os.path.exists(model_path):
            return jsonify({"error": "Model not found"}), 404
        keywords = extract_keywords(model_path, topn=topn)
        return jsonify({"keywords": keywords})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
