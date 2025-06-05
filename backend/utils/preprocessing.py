import re
import os
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec, KeyedVectors

# Ініціалізація лематизатора та стоп-слів
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Фрази для об'єднання через нижню риску
phrases_to_combine = {
    "machine learning": "machine_learning",
    "artificial intelligence": "artificial_intelligence",
    "natural language processing": "natural_language_processing",
}

def preprocess_text(file_path, final_path, final_file):

    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Перетворення в нижній регістр
    text = text.lower()

    # Замінюємо фрази на поєднання через нижню риску
    for phrase, combined in phrases_to_combine.items():
        text = text.replace(phrase, combined)

    # Видаляємо небуквені символи, окрім пробілів і точок
    text = re.sub(r'[^a-z\s\._]', '', text)

    # Видаляємо зайві пробіли
    text = re.sub(r'\s+', ' ', text).strip()

    # Лематизація і видалення стоп-слів
    words = text.split()
    processed_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    # Об'єднання слів у фінальний текст
    processed_text = ' '.join(processed_words)

    # Додаємо точки між реченнями (якщо необхідно)
    processed_text = re.sub(r'(\s*\.\s*)+', '. ', processed_text).strip()

    # Перетворення тексту у список слів для моделі Word2Vec
    sentences = [sentence.split() for sentence in processed_text.split('. ') if sentence]

    # Створення моделі Word2Vec
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

    # Перевірка та створення папки final_model
    os.makedirs(final_path, exist_ok=True)

    # Збереження KeyedVectors у форматі .kv
    base_name = os.path.splitext(final_file)[0]
    model.wv.save(f"{final_path}/{base_name}.kv")
    print(f"Векторизована модель збережена у: {final_path}/{final_file}.kv")
    
    return f"{final_path}/{final_file}.kv"

def extract_keywords(model_path, topn=10):
    """
    Повертає topn ключових слів, найближчих до середнього вектора всіх слів у моделі.
    """
    model = KeyedVectors.load(model_path)
    vocab = list(model.index_to_key)
    mean_vector = sum(model[word] for word in vocab) / len(vocab)
    similar = model.similar_by_vector(mean_vector, topn=topn)
    keywords = [word for word, score in similar]
    return keywords

# Виклик функції з прикладом
if __name__ == "__main__":
    input_file = "../first_model/gameofthrones.txt"  # Вказуємо шлях до вашого файлу
    final_path = "../final_model"
    final_file = "vectors"
    preprocess_text(input_file, final_path, final_file)