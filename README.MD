# Word2Vec Explorer

## Overview
A web application for exploring word embeddings using Word2Vec. This tool provides functionalities for word similarity, analogies, and visualizing word clusters.

## Description
Word2Vec Explorer allows users to interactively explore the relationships between words using Word2Vec models. The application is built with a Flask backend and a React frontend. Users can input words and view their similarities, find analogies, and visualize word clusters using PCA and KMeans.

## Getting Started

### Dependencies
- Python 3.8+
- Flask
- Word2Vec
- React
- Node.js
- Additional Python libraries: scikit-learn, matplotlib, base64, networkx

### Installing
1. Clone the repository:
   ```bash
   git clone https://github.com/DmytroBanar/word2vec.git

### Executing program
1. ```bash
    cd frontend_react
    npm install
    npm run build
    ...
    cd frontend_react/word2vec-explorer
    npm start

2. ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install Flask gensim scikit-learn matplotlib base64 networkx
    ...
    cd backend
    python app.py


### Authors
Dmytro Banar
