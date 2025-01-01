# Book Recommendation System

A book recommendation system built using **Streamlit**, **SentenceTransformers**, **Faiss**, and **pandas**. This system suggests books based on user preferences and provides various features like search functionality, random book suggestions, similar book recommendations, and data visualizations.

## Features

- **Search Books**: Allows users to search for books by title or author.
- **Find Similar Books**: Recommends similar books based on a selected book using advanced NLP (Sentence Transformers) and Faiss for efficient similarity search.
- **Get Random Book**: Provides a random book suggestion from the dataset.
- **View Favorites**: Users can add books to their favorites list and view them.
- **Visualizations**: Displays visualizations such as the distribution of book ratings and word clouds of authors.

## Tech Stack

- **Streamlit**: Framework for building the interactive web app.
- **Sentence Transformers**: Used to generate sentence embeddings for books and calculate similarities.
- **Faiss**: A library for efficient similarity search, used to speed up the recommendation process.
- **pandas**: For handling and processing the dataset.
- **matplotlib & seaborn**: For generating visualizations.

## Installation

To run the project locally, follow these steps:

1. **Clone the repository/Download the repository:**

   git clone https://github.com/DipenGohil/Book-Recommendation-System.git
   cd book-recommendation-system

2. **Install the required libraries:**

    pip install pandas streamlit faiss-cpu sentence-transformers matplotlib seaborn wordcloud

3. **Run the Streamlit app:**

    streamlit run app.py

## Usage
Once the app is running, you will see the following options in the sidebar:

- **Search Books**: Enter a keyword (title or author) to search for books.
- **Find Similar Books**: Select a book title to get similar book recommendations.
- **Get Random Book**: Click "Surprise Me!" for a random book suggestion.
- **View Favorites**: View the books you have added to your favorites list.
- **Visualizations**: View visualizations like rating distribution and author word clouds.
