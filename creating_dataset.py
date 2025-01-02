import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import traceback
try:
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")
except Exception as e:
    st.error("Error loading datasets. Please check your files.")
    st.stop()

# Filter popular movies
movie_rating_counts = ratings['movieId'].value_counts()
popular_movies = movie_rating_counts[movie_rating_counts >= 50].index

# Filter active users
user_rating_counts = ratings['userId'].value_counts()
active_users = user_rating_counts[user_rating_counts >= 20].index

# Apply combined filters
ratings_filtered = ratings[ratings['movieId'].isin(popular_movies) & ratings['userId'].isin(active_users)]
movies_filtered = movies[movies['movieId'].isin(ratings_filtered['movieId'].unique())]

# Sample the dataset
ratings_sampled = ratings_filtered.sample(n=1000000, random_state=42)
movies_sampled = movies[movies['movieId'].isin(ratings_sampled['movieId'].unique())]
ratings_with_genres = ratings_sampled.merge(movies_sampled[['movieId', 'genres']], on='movieId', how='left')

ratings_sampled.to_csv("ratings_reduced.csv", index=False)
movies_sampled.to_csv("movies_reduced.csv", index=False)
print("movies shape",movies_sampled.shape)
print("ratings_sampled shape",ratings_sampled.shape)
ratings_with_genres.to_csv("ratings_with_genres.csv", index=False)
