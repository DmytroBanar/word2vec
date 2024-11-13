from flask import Flask, request, jsonify
from gensim.models import KeyedVectors
from flask_cors import CORS
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import base64
from io import BytesIO
import numpy as np
from matplotlib.lines import Line2D
import networkx as nx  #

app = Flask(__name__)
CORS(app)

# Завантаження моделі Word2Vec

model = KeyedVectors.load("first_model/vectors.kv") #Модель на базі "Lucifer"
#model = KeyedVectors.load("final_model/vectors.kv") #Модель на базі "Game of Thrones"

@app.route('/similarity', methods=['GET'])
def similarity():
    word = request.args.get('word')
    num_neighbors = int(request.args.get('num_neighbors', 10))
    
    print(f"Received word: {word}, num_neighbors: {num_neighbors}")  # Логування для перевірки параметрів
    
    similar_words = model.most_similar(word, topn=num_neighbors)
    
    print(f"Similar words found: {similar_words}")  # Логування результату пошуку
    
    return jsonify(similar_words)


@app.route('/anology', methods=['POST'])
def antology():
    # Встановлення значень за замовчуванням
    word = None
    pair = []
    num_analogies = 10  # Кількість аналогій за замовчуванням

    data = request.json
    word = data.get('word')
    pair = data.get('pair', [])
    num_analogies = data.get('num_analogies', num_analogies)

    # Перевірка необхідних полів
    if not word or len(pair) != 2:
        return jsonify({"error": "Будь ласка, надайте 'word' та пару з точно двох елементів."}), 400

    # Виконання функції аналогії моделі
    try:
        analogy_words = model.most_similar(positive=[word, pair[1]], negative=[pair[0]], topn=num_analogies)
        return jsonify(analogy_words)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def plot_word_clusters_base64(model, words, num_clusters=5):
    # Отримуємо вектори слів безпосередньо з model
    word_vectors = [model[word] for word in words if word in model]
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(word_vectors)
    labels = kmeans.labels_

    # Застосування PCA для зменшення до 2 вимірів
    pca = PCA(n_components=2)
    word_vectors_pca = pca.fit_transform(word_vectors)
    
    # Створення графіку
    plt.figure(figsize=(10, 10))
    for i, word in enumerate(words):
        if word in model:
            plt.scatter(word_vectors_pca[i, 0], word_vectors_pca[i, 1], label=labels[i])
            plt.annotate(word, (word_vectors_pca[i, 0], word_vectors_pca[i, 1]))

    # Збереження зображення у буфер
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    
    # Конвертування зображення в base64
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return img_base64

@app.route('/plot_similar_words_clusters', methods=['GET'])
def plot_similar_words_clusters():
    word = request.args.get('word')
    num_neighbors = int(request.args.get('num_neighbors', 10))
    num_clusters = int(request.args.get('num_clusters', 5))
    
    if not word:
        return jsonify({"error": "Please provide a 'word' parameter."}), 400

    try:
        # Отримуємо найближчі слова
        similar_words = model.most_similar(word, topn=num_neighbors)
        words = [w[0] for w in similar_words]  # Список слів
        word_vectors = []

        # Перевірка наявності слів у моделі
        for word in words:
            if word in model:
                word_vectors.append(model[word])
            else:
                print(f"Word '{word}' not found in the model.")  # Логування відсутніх слів
        
        # Якщо немає векторів для слів, повертаємо помилку
        if not word_vectors:
            return jsonify({"error": "No valid words found in the model."}), 400
        
        # Кластеризація слів
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(word_vectors)
        labels = kmeans.labels_

        # Застосування PCA для зменшення до 2 вимірів
        pca = PCA(n_components=2)
        word_vectors_pca = pca.fit_transform(word_vectors)
        
        # Створення графіку
        plt.figure(figsize=(10, 10))

        # Розмір точок залежно від величини векторів
        max_norm = max(np.linalg.norm(model[word]) for word in words)
        point_sizes = [10 + 100 * (np.linalg.norm(model[word]) / max_norm) for word in words]

        # Для кожного кластера обираємо свій колір
        cluster_colors = plt.cm.viridis(np.linspace(0, 1, num_clusters))
        
        # Створення графіку для кожного слова, визначаємо колір по кластеру
        for i, word in enumerate(words):
            color = cluster_colors[labels[i]]  # Кожному кластеру свій колір
            size = point_sizes[i] * 10  # Збільшення розміру для візуалізації
            plt.scatter(word_vectors_pca[i, 0], word_vectors_pca[i, 1], color=color, s=size, alpha=0.7)
            plt.annotate(word, (word_vectors_pca[i, 0], word_vectors_pca[i, 1]))

        # Додавання сітки
        plt.grid(True)

        # Обчислення відсотка слів для кожного кластеру
        cluster_counts = [np.sum(labels == i) for i in range(num_clusters)]
        total_words = len(words)
        cluster_percentages = [count / total_words * 100 for count in cluster_counts]

        # Додавання легенди з відсотками
        legend_labels = [f'{cluster_percentages[i]:.1f}%' for i in range(num_clusters)]
        legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=cluster_colors[i], markersize=10, label=legend_labels[i]) for i in range(num_clusters)]
        plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))

        # Збереження зображення у буфер
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)

        # Конвертування зображення в base64
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return jsonify({"image": img_base64})
    
    except Exception as e:
        print(f"Error: {str(e)}")  # Логування помилки
        return jsonify({"error": str(e)}), 500

@app.route('/plot_character_connections', methods=['GET'])
def plot_character_connections():
    # Отримуємо список персонажів або об'єктів
    characters = request.args.get('characters')
    if not characters:
        return jsonify({"error": "Please provide 'characters' parameter."}), 400

    # Логування отриманих значень
    print(f"Received characters: {characters}")

    # Розділяємо строки через кому і очищуємо зайві пробіли
    characters = [char.strip() for char in characters.split(',')]
    
    # Логування після обробки
    print(f"Processed characters: {characters}")

    # Перевірка наявності слів в моделі
    valid_characters = [char for char in characters if char in model]
    if not valid_characters:
        return jsonify({"error": "No valid characters found in the model."}), 400

    try:
        # Створення графа
        G = nx.Graph()

        # Додавання вузлів для персонажів
        for char in valid_characters:
            G.add_node(char)

        # Додавання зв'язків між персонажами
        for i, char1 in enumerate(valid_characters):
            for char2 in valid_characters[i+1:]:
                similarity = model.similarity(char1, char2)  # Обчислюємо схожість між персонажами
                if similarity > 0.5:  # Якщо схожість велика, додаємо зв'язок
                    G.add_edge(char1, char2, weight=similarity)

        # Малювання графу
        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(G)  # Розміщення вузлів
        nx.draw_networkx_nodes(G, pos, node_size=700)
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
        nx.draw_networkx_edges(G, pos, width=2, alpha=0.5, edge_color='b')
        plt.title("Словесні зв'язки між персонажами та об'єктами")

        # Додавання відстаней між зв'язками (схожість)
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

        # Збереження зображення у буфер
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)

        # Конвертування зображення в base64
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return jsonify({"image": img_base64})
    
    except Exception as e:
        # Логування помилки
        print(f"Error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
