# Amazon Sentiment Analysis NLP

## Overview
This repository contains a collection of Natural Language Processing (NLP) projects focused on sentiment analysis and text preprocessing techniques. The main project analyzes Amazon customer feedback using sentiment analysis and a variety of NLP techniques such as stemming, lemmatization, tokenization, and vectorization. The projects utilize popular NLP libraries like `spaCy`, `NLTK`, and `Scikit-learn` for tokenization, feature extraction, and model building.

## Project Files
- **Amazon_feedback_sentimental_analysis.ipynb**: The main notebook where sentiment analysis is performed on Amazon customer feedback. It explores data preprocessing techniques, feature extraction, and machine learning models.
- **Stemming_Lemmatization.ipynb**: Demonstrates stemming and lemmatization techniques using `NLTK` and `spaCy` to normalize text data.
- **Vectorizer_and_Simple_SVM.ipynb**: Focuses on feature extraction using `CountVectorizer`, `TfidfVectorizer`, and building a simple Support Vector Machine (SVM) model for text classification.
- **Word_Vectors.ipynb**: Covers word embedding techniques like Word2Vec and explores how word vectors can be used in NLP tasks.
- **regex.ipynb**: A notebook demonstrating the use of regular expressions for text preprocessing and extraction.
- **spacy_nltk_tokenization.ipynb**: Shows tokenization techniques using both `spaCy` and `NLTK`, comparing their capabilities in handling different text structures.

## Key Features
- **Text Preprocessing**: Includes techniques such as stemming, lemmatization, tokenization, and regular expressions to clean and prepare text data for analysis.
- **Feature Extraction**: Utilizes `CountVectorizer`, `TfidfVectorizer`, and word embeddings to convert text into numeric formats suitable for machine learning models.
- **Machine Learning**: Applies a simple Support Vector Machine (SVM) for sentiment classification and explores its performance on preprocessed Amazon customer feedback.
- **Word Embeddings**: Implements Word2Vec to represent text as vectors and demonstrates how these representations enhance model performance.

## Prerequisites
Before running the notebooks, ensure you have the following dependencies installed:
- Python 3.x
- pandas
- numpy
- scikit-learn
- NLTK
- spaCy
- matplotlib
- seaborn

You can install the necessary dependencies using the following command:
`
pip install pandas numpy scikit-learn nltk spacy matplotlib seaborn`

## How to Run
Clone the repository:
`git clone https://github.com/yourusername/Amazon-Sentiment-Analysis-NLP.git
cd Amazon-Sentiment-Analysis-NLP`

Open the notebooks in Jupyter or Google Colab to execute the code and explore the sentiment analysis and text preprocessing techniques.
