import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os

# Load dataset
books = pd.read_csv('books-dataset.csv')

# Preprocess Data
books['combined_features'] = (
    books['title'].fillna('') + ' ' +
    books['authors'].fillna('') + ' ' +
    books['language_code'].fillna('')
)

# Cache SentenceTransformer model using @st.cache_resource
@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Load pre-trained model
model = load_model()

# Cache precomputed embeddings if they exist, otherwise compute them
embedding_cache_path = 'embeddings.npy'

if os.path.exists(embedding_cache_path):
    embeddings = np.load(embedding_cache_path)
else:
    embeddings = model.encode(books['combined_features'].tolist())
    np.save(embedding_cache_path, embeddings)

# Initialize Faiss index for efficient similarity search
index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance (cosine similarity)
index.add(embeddings)

# Initialize favorites in session state
if 'favorites' not in st.session_state:
    st.session_state.favorites = []

# Helper Functions
def search_books(query):
    """Search books by title, author, or publisher."""
    query_lower = query.lower()
    results = books[
        books['title'].str.contains(query_lower, case=False, na=False) |
        books['authors'].str.contains(query_lower, case=False, na=False)
    ]
    return results[['title', 'authors', 'average_rating', 'language_code']]

def get_similar_books(book_title, top_n=5):
    """Recommend books based on similarity using Faiss."""
    if book_title not in books['title'].values:
        return "Book not found in the dataset."
    idx = books[books['title'] == book_title].index[0]
    embedding_query = embeddings[idx].reshape(1, -1)
    
    distances, indices = index.search(embedding_query, top_n + 1)  # Get top_n + 1 results (excluding the book itself)
    book_indices = indices.flatten()[1:]  # Exclude the book itself
    return books.iloc[book_indices][['title', 'authors', 'average_rating', 'language_code']]

def get_random_book():
    """Fetch a random book."""
    random_idx = random.randint(0, len(books) - 1)
    return books.iloc[random_idx][['title', 'authors', 'average_rating', 'language_code']]

def add_to_favorites(book_title):
    """Add a book to the user's favorites list."""
    if book_title not in st.session_state.favorites:
        st.session_state.favorites.append(book_title)

def visualize_data():
    """Generate visualizations."""
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    # Ratings Distribution
    sns.histplot(books['average_rating'], bins=20, kde=True, ax=ax[0])
    ax[0].set_title('Distribution of Average Ratings')
    ax[0].set_xlabel('Average Rating')
    
    # Word Cloud for Authors
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(books['authors']))
    ax[1].imshow(wordcloud, interpolation='bilinear')
    ax[1].axis('off')
    ax[1].set_title('Word Cloud of Authors')
    st.pyplot(fig)

# Streamlit App
st.title("Book Recommendation System")

option = st.sidebar.radio("Choose an option:", [
    "Search Books",
    "Find Similar Books",
    "Get Random Book",
    "View Favorites",
    "Visualizations"
])

if option == "Search Books":
    st.header("Search Books")
    query = st.text_input("Enter a keyword (title/author):")
    if query:
        results = search_books(query)
        if not results.empty:
            st.table(results)
        else:
            st.error("No results found.")

elif option == "Find Similar Books":
    st.header("Find Similar Books")
    book_title = st.selectbox("Choose a book title:", books['title'])
    top_n = st.slider("Number of recommendations:", min_value=1, max_value=20, value=5)
    if st.button("Get Recommendations"):
        similar_books = get_similar_books(book_title, top_n)
        if isinstance(similar_books, str):
            st.error(similar_books)
        else:
            for _, row in similar_books.iterrows():
                st.write(f"**{row['title']}** by {row['authors']} (Rating: {row['average_rating']})")
                st.button("Add to Favorites", key=row['title'], on_click=add_to_favorites, args=(row['title'],))

elif option == "Get Random Book":
    st.header("Random Book Suggestion")
    if st.button("Surprise Me!"):
        random_book = get_random_book()
        st.write(f"**{random_book['title']}** by {random_book['authors']} (Rating: {random_book['average_rating']})")
        st.button("Add to Favorites", on_click=add_to_favorites, args=(random_book['title'],))

elif option == "View Favorites":
    st.header("Your Favorites")
    if st.session_state.favorites:
        favorite_books = books[books['title'].isin(st.session_state.favorites)]
        st.table(favorite_books[['title', 'authors', 'average_rating', 'language_code']])
    else:
        st.info("Your favorites list is empty.")

elif option == "Visualizations":
    st.header("Data Visualizations")
    visualize_data()
