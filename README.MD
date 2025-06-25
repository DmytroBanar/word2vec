# Web application for processing and semantic analysis of text data

## Overview
A web application for exploring word embeddings using Word2Vec. This tool provides functionalities for word similarity, analogies, and visualizing word clusters.
Users can upload their own text files, train custom models, analyze relationships between words, and interactively visualize results through an intuitive web interface.
## Description
Word2Vec Explorer allows users to interactively explore the relationships between words using Word2Vec models. The application is built with a Flask backend and a React frontend. Users can upload their own text files, train custom models, input words to view their similarities, find analogies, and visualize word clusters using PCA and KMeans. The system also supports user authentication, management of personal and default models, and provides interactive analytics and visualizations for deeper text analysis.

## Getting Started

### Dependencies
- Python 3.8+
- Flask
- Gensim (Word2Vec)
- scikit-learn
- matplotlib
- networkx
- python-dotenv
- flask-jwt-extended
- flask-mail
- mysql-connector-python (або sqlite3 для тестування)
- React
- Node.js
- (Frontend) react-chartjs-2, chart.js, d3.js

### Installing
1. Clone the repository:
   ```bash
   git clone https://github.com/DmytroBanar/word2vec.git

### Executing program
1. ```bashcd frontend_react
    ```bashnpm install
    ```bashnpm run build
    ...
    ```bashcd frontend_react/word2vec-explorer
    ```bashnpm start

2. ```bashpython3 -m venv venv
    ```bashsource venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```bashpip install Flask gensim scikit-learn matplotlib networkx python-dotenv flask-jwt-extended flask-mail mysql-connector-python
    ...
    ```bashcd backend
    ```bashpython app.py


### Authors
Dmytro Banar
