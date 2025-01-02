# import pandas as pd
# import requests
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from surprise import Dataset, Reader, SVD
# from surprise.model_selection import train_test_split
# import streamlit as st
# import traceback

# # Function to fetch poster from TMDb
# def fetch_poster(tmdb_id):
#     try:
#         response = requests.get(
#             f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key=b04f9dd95202a019a4dd1dc7aa07e7f5&language=en-US"
#         )
#         if response.status_code == 200:
#             data = response.json()
#             return f"https://image.tmdb.org/t/p/w500{data['poster_path']}"
#         else:
#             return "https://via.placeholder.com/150"  # Placeholder for missing posters
#     except Exception as e:
#         return "https://via.placeholder.com/150"  # Fallback for errors

# # Load datasets
# try:
#     movies = pd.read_csv("movies_reduced.csv")
#     ratings = pd.read_csv("ratings_reduced.csv")
#     links = pd.read_csv("links.csv")  # Contains movieId and tmdbId mapping
# except Exception as e:
#     st.error("Error loading datasets. Please check your files.")
#     st.stop()

# # Merge movies with links to include tmdbId
# movies = movies.merge(links[['movieId', 'tmdbId']], on='movieId', how='left')

# # Preprocess genres
# movies['genres'] = movies['genres'].fillna('').str.replace('|', ' ')

# # Create a TF-IDF matrix for genres
# try:
#     tfidf = TfidfVectorizer(stop_words='english')
#     tfidf_matrix = tfidf.fit_transform(movies['genres'])
#     content_similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)
# except Exception as e:
#     st.error(f"Error during TF-IDF processing: {traceback.format_exc()}")
#     st.stop()

# # Prepare data for collaborative filtering
# try:
#     reader = Reader(rating_scale=(0.5, 5.0))
#     data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
#     trainset, testset = train_test_split(data, test_size=0.2)
#     model = SVD()
#     model.fit(trainset)
# except Exception as e:
#     st.error(f"Error during collaborative filtering setup: {traceback.format_exc()}")
#     st.stop()

# # Function for content-based recommendations
# def recommend_content(movie_title, top_n=10):
#     try:
#         if movie_title not in movies['title'].values:
#             return []
#         idx = movies[movies['title'] == movie_title].index[0]
#         sim_scores = list(enumerate(content_similarity[idx]))
#         sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
#         top_movies = [
#             (
#                 movies.iloc[i[0]].title,
#                 fetch_poster(movies.iloc[i[0]]['tmdbId'])
#             )
#             for i in sim_scores[1:top_n+1]
#         ]
#         return top_movies
#     except Exception as e:
#         st.error(f"Error in content recommendation: {traceback.format_exc()}")
#         return []

# # Function for hybrid recommendations
# def hybrid_recommendation(user_id, movie_title, top_n=10):
#     try:
#         content_recs = recommend_content(movie_title, top_n=top_n)
#         if not content_recs:
#             return []
#         content_indices = movies[movies['title'].isin([rec[0] for rec in content_recs])].index
#         collab_scores = [
#             (movie, url, model.predict(user_id, movies.iloc[idx]['movieId']).est)
#             for (movie, url), idx in zip(content_recs, content_indices)
#         ]
#         collab_scores = sorted(collab_scores, key=lambda x: x[2], reverse=True)
#         return [(movie, url) for movie, url, score in collab_scores[:top_n]]
#     except Exception as e:
#         st.error(f"Error in hybrid recommendation: {traceback.format_exc()}")
#         return []

# # Streamlit app
# st.title("Hybrid Movie Recommender System")
# st.write("Combining content-based and collaborative filtering for movie recommendations!")

# # User input
# try:
#     user_id = st.number_input("Enter User ID", min_value=1, value=1, step=1)
#     movie_title = st.selectbox("Select a Movie Title", movies['title'].unique())
#     top_n = st.slider("Number of Recommendations", min_value=1, max_value=20, value=10)

#     # Recommendations
#     if st.button("Get Recommendations"):
#         recommendations = hybrid_recommendation(user_id, movie_title, top_n)
#         if recommendations:
#             st.subheader("Recommended Movies:")

#             # Create tiles (5 movies in a row)
#             cols = st.columns(5)  # Create 5 columns for tiles
#             for idx, (movie, poster_url) in enumerate(recommendations):
#                 col = cols[idx % 5]  # Cycle through columns
#                 with col:
#                     st.image(poster_url, width=120)
#                     st.caption(movie)
#         else:
#             st.write("No recommendations found. Try a different movie!")
# except Exception as e:
#     st.error(f"Error in Streamlit app: {traceback.format_exc()}")


# import pandas as pd
# import requests
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from surprise import Dataset, Reader, SVD
# from surprise.model_selection import train_test_split
# import streamlit as st
# import traceback

# # Function to fetch poster from TMDb
# def fetch_poster(tmdb_id):
#     try:
#         response = requests.get(
#             f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key=b04f9dd95202a019a4dd1dc7aa07e7f5&language=en-US"
#         )
#         if response.status_code == 200:
#             data = response.json()
#             return f"https://image.tmdb.org/t/p/w500{data['poster_path']}"
#         else:
#             return "https://via.placeholder.com/150"  # Placeholder for missing posters
#     except Exception as e:
#         return "https://via.placeholder.com/150"  # Fallback for errors

# # Load datasets
# try:
#     movies = pd.read_csv("movies_reduced.csv")
#     ratings = pd.read_csv("ratings_reduced.csv")
#     links = pd.read_csv("links.csv")  # Contains movieId and tmdbId mapping
# except Exception as e:
#     st.error("Error loading datasets. Please check your files.")
#     st.stop()

# # Merge movies with links to include tmdbId
# movies = movies.merge(links[['movieId', 'tmdbId']], on='movieId', how='left')

# # Preprocess genres
# movies['genres'] = movies['genres'].fillna('').str.replace('|', ' ')

# # Create a TF-IDF matrix for genres
# try:
#     tfidf = TfidfVectorizer(stop_words='english')
#     tfidf_matrix = tfidf.fit_transform(movies['genres'])
#     content_similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)
# except Exception as e:
#     st.error(f"Error during TF-IDF processing: {traceback.format_exc()}")
#     st.stop()

# # Prepare data for collaborative filtering
# try:
#     reader = Reader(rating_scale=(0.5, 5.0))
#     data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
#     trainset, testset = train_test_split(data, test_size=0.2)
#     model = SVD()
#     model.fit(trainset)
# except Exception as e:
#     st.error(f"Error during collaborative filtering setup: {traceback.format_exc()}")
#     st.stop()

# # Ensure ratings and movies are merged to include genres
# ratings_with_genres = ratings.merge(movies[['movieId', 'genres']], on='movieId', how='left')

# # Function to get user's top genres
# def get_user_top_genres(user_id):
#     try:
#         # Filter ratings for the specific user
#         user_ratings = ratings_with_genres[ratings_with_genres['userId'] == user_id]

#         # Handle case where no ratings are found for the user
#         if user_ratings.empty:
#             st.warning("No ratings found for this user. Please try a different user ID.")
#             return []

#         # Calculate genre counts from the user's ratings
#         genre_counts = user_ratings['genres'].str.split(' ').explode().value_counts()
#         top_genres = genre_counts.head(2).index.tolist()  # Top 2 genres
#         return top_genres
#     except Exception as e:
#         st.error(f"Error in calculating user's top genres: {traceback.format_exc()}")
#         return []

# # Function to recommend top movies from a genre
# def recommend_top_movies_from_genre(genre, top_n=10):
#     try:
#         # Filter movies by genre (case-insensitive and optimized)
#         genre_movies = movies[movies['genres'].str.contains(rf'\b{genre}\b', case=False, na=False)]

#         # Select only the necessary columns and sort by popularity if available
#         genre_movies = genre_movies[['movieId', 'tmdbId', 'title']].head(top_n)

#         # Fetch posters only for the top N movies
#         genre_movies['poster_url'] = genre_movies['tmdbId'].apply(fetch_poster)

#         return genre_movies
#     except Exception as e:
#         st.error(f"Error in genre-based recommendation: {traceback.format_exc()}")
#         return pd.DataFrame()


# # Function for content-based recommendations
# def recommend_content(movie_title, top_n=10):
#     try:
#         if movie_title not in movies['title'].values:
#             return []
#         idx = movies[movies['title'] == movie_title].index[0]
#         sim_scores = list(enumerate(content_similarity[idx]))
#         sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
#         top_movies = [
#             (
#                 movies.iloc[i[0]].title,
#                 fetch_poster(movies.iloc[i[0]]['tmdbId'])
#             )
#             for i in sim_scores[1:top_n+1]
#         ]
#         return top_movies
#     except Exception as e:
#         st.error(f"Error in content recommendation: {traceback.format_exc()}")
#         return []

# # Function for hybrid recommendations
# def hybrid_recommendation(user_id, movie_title, top_n=10):
#     try:
#         content_recs = recommend_content(movie_title, top_n=top_n)
#         if not content_recs:
#             return []
#         content_indices = movies[movies['title'].isin([rec[0] for rec in content_recs])].index
#         collab_scores = [
#             (movie, url, model.predict(user_id, movies.iloc[idx]['movieId']).est)
#             for (movie, url), idx in zip(content_recs, content_indices)
#         ]
#         collab_scores = sorted(collab_scores, key=lambda x: x[2], reverse=True)
#         return [(movie, url) for movie, url, score in collab_scores[:top_n]]
#     except Exception as e:
#         st.error(f"Error in hybrid recommendation: {traceback.format_exc()}")
#         return []

# # Streamlit app
# # Streamlit app
# st.title("Hybrid Movie Recommender System")
# st.write("Combining content-based and collaborative filtering for movie recommendations!")

# try:
#     # User input
#     # user_id = st.number_input("Enter User ID", min_value=1, value=1, step=1)
#     user_id = st.selectbox("Enter User ID", ratings['userId'].unique())
#     movie_title = st.selectbox("Select a Movie Title", movies['title'].unique())
#     top_n = st.slider("Number of Recommendations", min_value=1, max_value=20, value=10)

#     # Hybrid recommendations
#     if st.button("Get Recommendations"):
#         recommendations = hybrid_recommendation(user_id, movie_title, top_n)

#         # Display hybrid recommendations
#         if recommendations:
#             st.subheader("Recommended Movies:")
#             cols = st.columns(5)  # Create 5 columns for tiles
#             for idx, (movie, poster_url) in enumerate(recommendations):
#                 col = cols[idx % 5]  # Cycle through columns
#                 with col:
#                     st.image(poster_url, width=120)
#                     st.caption(movie)
#         else:
#             st.write("No recommendations found. Try a different movie!")

#         # Top genres recommendations
#         top_genres = get_user_top_genres(user_id)
#         if top_genres:
#             for genre in top_genres:
#                 st.subheader(f"Top 10 Movies in {genre} Genre:")
#                 top_movies = recommend_top_movies_from_genre(genre)

#                 # Display top movies for the genre
#                 genre_cols = st.columns(5)  # 5 tiles per row
#                 for idx, row in top_movies.iterrows():
#                     col = genre_cols[idx % 5]  # Cycle through columns
#                     with col:
#                         st.image(row['poster_url'], width=120)
#                         st.caption(row['title'])
# except Exception as e:
#     st.error(f"Error in Streamlit app: {traceback.format_exc()}")







# import pandas as pd
# import requests
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from surprise import Dataset, Reader, SVD
# from surprise.model_selection import train_test_split
# import streamlit as st
# import traceback

# # Function to fetch poster from TMDb
# def fetch_poster(tmdb_id):
#     try:
#         response = requests.get(
#             f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key=b04f9dd95202a019a4dd1dc7aa07e7f5&language=en-US"
#         )
#         if response.status_code == 200:
#             data = response.json()
#             return f"https://image.tmdb.org/t/p/w500{data['poster_path']}"
#         else:
#             return "https://via.placeholder.com/150"  # Placeholder for missing posters
#     except Exception as e:
#         return "https://via.placeholder.com/150"  # Fallback for errors

# # Load datasets
# try:
#     movies = pd.read_csv("movies_reduced.csv")
#     ratings = pd.read_csv("ratings_reduced.csv")
#     links = pd.read_csv("links.csv")  # Contains movieId and tmdbId mapping
# except Exception as e:
#     st.error("Error loading datasets. Please check your files.")
#     st.stop()

# # Merge movies with links to include tmdbId
# movies = movies.merge(links[['movieId', 'tmdbId']], on='movieId', how='left')

# # Preprocess genres
# movies['genres'] = movies['genres'].fillna('').str.replace('|', ' ')

# # Create a TF-IDF matrix for genres
# try:
#     tfidf = TfidfVectorizer(stop_words='english')
#     tfidf_matrix = tfidf.fit_transform(movies['genres'])
#     content_similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)
# except Exception as e:
#     st.error(f"Error during TF-IDF processing: {traceback.format_exc()}")
#     st.stop()

# # Prepare data for collaborative filtering
# try:
#     reader = Reader(rating_scale=(0.5, 5.0))
#     data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
#     trainset, testset = train_test_split(data, test_size=0.2)
#     model = SVD()
#     model.fit(trainset)
# except Exception as e:
#     st.error(f"Error during collaborative filtering setup: {traceback.format_exc()}")
#     st.stop()

# # Ensure ratings and movies are merged to include genres
# ratings_with_genres = ratings.merge(movies[['movieId', 'genres']], on='movieId', how='left')

# # Function to get user's top genres
# def get_user_top_genres(user_id):
#     try:
#         # Filter ratings for the specific user
#         user_ratings = ratings_with_genres[ratings_with_genres['userId'] == user_id]

#         # Handle case where no ratings are found for the user
#         if user_ratings.empty:
#             st.warning("No ratings found for this user. Please try a different user ID.")
#             return []

#         # Calculate genre counts from the user's ratings
#         genre_counts = user_ratings['genres'].str.split(' ').explode().value_counts()
#         top_genres = genre_counts.head(2).index.tolist()  # Top 2 genres
#         return top_genres
#     except Exception as e:
#         st.error(f"Error in calculating user's top genres: {traceback.format_exc()}")
#         return []

# # Function to recommend top movies from a genre
# def recommend_top_movies_from_genre(genre, top_n=10):
#     try:
#         # Filter movies by genre (case-insensitive and optimized)
#         genre_movies = movies[movies['genres'].str.contains(rf'\b{genre}\b', case=False, na=False)]

#         # Select only the necessary columns and sort by popularity if available
#         genre_movies = genre_movies[['movieId', 'tmdbId', 'title']].head(top_n)

#         # Fetch posters only for the top N movies
#         genre_movies['poster_url'] = genre_movies['tmdbId'].apply(fetch_poster)

#         return genre_movies
#     except Exception as e:
#         st.error(f"Error in genre-based recommendation: {traceback.format_exc()}")
#         return pd.DataFrame()


# # Function for content-based recommendations
# def recommend_content(movie_title, top_n=10):
#     try:
#         if movie_title not in movies['title'].values:
#             return []
#         idx = movies[movies['title'] == movie_title].index[0]
#         sim_scores = list(enumerate(content_similarity[idx]))
#         sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
#         top_movies = [
#             (
#                 movies.iloc[i[0]].title,
#                 fetch_poster(movies.iloc[i[0]]['tmdbId'])
#             )
#             for i in sim_scores[1:top_n+1]
#         ]
#         return top_movies
#     except Exception as e:
#         st.error(f"Error in content recommendation: {traceback.format_exc()}")
#         return []

# # Function for hybrid recommendations
# def hybrid_recommendation(user_id, movie_title, top_n=10):
#     try:
#         content_recs = recommend_content(movie_title, top_n=top_n)
#         if not content_recs:
#             return []
#         content_indices = movies[movies['title'].isin([rec[0] for rec in content_recs])].index
#         collab_scores = [
#             (movie, url, model.predict(user_id, movies.iloc[idx]['movieId']).est)
#             for (movie, url), idx in zip(content_recs, content_indices)
#         ]
#         collab_scores = sorted(collab_scores, key=lambda x: x[2], reverse=True)
#         return [(movie, url) for movie, url, score in collab_scores[:top_n]]
#     except Exception as e:
#         st.error(f"Error in hybrid recommendation: {traceback.format_exc()}")
#         return []

# # Streamlit app
# st.title("Hybrid Movie Recommender System")
# st.write("Combining content-based and collaborative filtering for movie recommendations!")

# try:
#     # User input
#     user_id = st.selectbox("Enter User ID", ratings['userId'].unique())
#     movie_title = st.selectbox("Select a Movie Title", movies['title'].unique())
#     top_n = st.slider("Number of Recommendations", min_value=1, max_value=20, value=10)

#     # Hybrid recommendations
#     if st.button("Get Recommendations"):
#         recommendations = hybrid_recommendation(user_id, movie_title, top_n)

#         # Display hybrid recommendations
#         if recommendations:
#             st.subheader("Recommended Movies:")
#             cols = st.columns(3)  # Create 5 columns for tiles
#             for idx, (movie, poster_url) in enumerate(recommendations):
#                 col = cols[idx % 3]  # Cycle through columns
#                 with col:
#                     st.image(poster_url, width=120)
#                     st.caption(movie)
#         else:
#             st.write("No recommendations found. Try a different movie!")

#         # Top genres recommendations
#         top_genres = get_user_top_genres(user_id)
#         if top_genres:
#             for genre in top_genres:
#                 st.subheader(f"Top 10 Movies in {genre} Genre:")
#                 top_movies = recommend_top_movies_from_genre(genre)

#                 # Display top movies for the genre
#                 genre_cols = st.columns(3)  # 5 tiles per row
#                 for idx, row in top_movies.iterrows():
#                     col = genre_cols[idx % 3]  # Cycle through columns
#                     with col:
#                         st.image(row['poster_url'], width=120)
#                         st.caption(row['title'])
# except Exception as e:
#     st.error(f"Error in Streamlit app: {traceback.format_exc()}")










import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import streamlit as st
import traceback

# Function to fetch poster from TMDb
def fetch_poster(tmdb_id):
    try:
        response = requests.get(
            f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key=b04f9dd95202a019a4dd1dc7aa07e7f5&language=en-US"
        )
        if response.status_code == 200:
            data = response.json()
            return f"https://image.tmdb.org/t/p/w500{data['poster_path']}"
        else:
            return "https://via.placeholder.com/150"  # Placeholder for missing posters
    except Exception as e:
        return "https://via.placeholder.com/150"  # Fallback for errors

# Load datasets
try:
    movies = pd.read_csv("movies_reduced.csv")
    print("movies shape",movies.shape)
    ratings = pd.read_csv("ratings_reduced.csv")
    print("ratings shape",ratings.shape)
    links = pd.read_csv("links.csv")  # Contains movieId and tmdbId mapping
    print("links shape",links.shape)
except Exception as e:
    st.error("Error loading datasets. Please check your files.")
    st.stop()

# Merge movies with links to include tmdbId
movies = movies.merge(links[['movieId', 'tmdbId']], on='movieId', how='left')

# Preprocess genres
movies['genres'] = movies['genres'].fillna('').str.replace('|', ' ')

# Create a TF-IDF matrix for genres
try:
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['genres'])
    content_similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)
except Exception as e:
    st.error(f"Error during TF-IDF processing: {traceback.format_exc()}")
    st.stop()

# Prepare data for collaborative filtering
try:
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.2)
    model = SVD()
    model.fit(trainset)
except Exception as e:
    st.error(f"Error during collaborative filtering setup: {traceback.format_exc()}")
    st.stop()

# Ensure ratings and movies are merged to include genres
ratings_with_genres = ratings.merge(movies[['movieId', 'genres']], on='movieId', how='left')

# Function to get user's top genres
def get_user_top_genres(user_id):
    try:
        # Filter ratings for the specific user
        user_ratings = ratings_with_genres[ratings_with_genres['userId'] == user_id]

        # Handle case where no ratings are found for the user
        if user_ratings.empty:
            st.warning("No ratings found for this user. Please try a different user ID.")
            return []

        # Calculate genre counts from the user's ratings
        genre_counts = user_ratings['genres'].str.split(' ').explode().value_counts()
        top_genres = genre_counts.head(2).index.tolist()  # Top 2 genres
        return top_genres
    except Exception as e:
        st.error(f"Error in calculating user's top genres: {traceback.format_exc()}")
        return []

# Function to recommend top movies from a genre
def recommend_top_movies_from_genre(genre, top_n=10):
    try:
        # Filter movies by genre (case-insensitive and optimized)
        genre_movies = movies[movies['genres'].str.contains(rf'\b{genre}\b', case=False, na=False)]

        # Select only the necessary columns and sort by popularity if available
        genre_movies = genre_movies[['movieId', 'tmdbId', 'title']].head(top_n)

        # Fetch posters only for the top N movies
        genre_movies['poster_url'] = genre_movies['tmdbId'].apply(fetch_poster)

        return genre_movies
    except Exception as e:
        st.error(f"Error in genre-based recommendation: {traceback.format_exc()}")
        return pd.DataFrame()


# Function for content-based recommendations
def recommend_content(movie_title, top_n=10):
    try:
        if movie_title not in movies['title'].values:
            return []
        idx = movies[movies['title'] == movie_title].index[0]
        sim_scores = list(enumerate(content_similarity[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        top_movies = [
            (
                movies.iloc[i[0]].title,
                fetch_poster(movies.iloc[i[0]]['tmdbId'])
            )
            for i in sim_scores[1:top_n+1]
        ]
        return top_movies
    except Exception as e:
        st.error(f"Error in content recommendation: {traceback.format_exc()}")
        return []

# Function for hybrid recommendations
def hybrid_recommendation(user_id, movie_title, top_n=10):
    try:
        content_recs = recommend_content(movie_title, top_n=top_n)
        if not content_recs:
            return []
        content_indices = movies[movies['title'].isin([rec[0] for rec in content_recs])].index
        collab_scores = [
            (movie, url, model.predict(user_id, movies.iloc[idx]['movieId']).est)
            for (movie, url), idx in zip(content_recs, content_indices)
        ]
        collab_scores = sorted(collab_scores, key=lambda x: x[2], reverse=True)
        return [(movie, url) for movie, url, score in collab_scores[:top_n]]
    except Exception as e:
        st.error(f"Error in hybrid recommendation: {traceback.format_exc()}")
        return []

# Streamlit app
st.title("Hybrid Movie Recommender System")
st.write("Combining content-based and collaborative filtering for movie recommendations!")

# CSS for fixed row height
st.markdown("""
    <style>
        .movie-tile {
            display: flex;
            flex-direction: column;
            align-items: center;
            
            height: 300px;  /* Fixed height */
            padding: 10px;
            text-align: center;
        }
        .movie-tile img {
            height: 150px;
            width: auto;
        }
        .movie-tile span {
            margin-top: 10px;
            font-size: 14px;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

try:
    # User input
    user_id = st.selectbox("Enter User ID", ratings['userId'].unique())
    movie_title = st.selectbox("Select a Movie Title", movies['title'].unique())
    top_n = st.slider("Number of Recommendations", min_value=1, max_value=20, value=10)

    # Hybrid recommendations
    if st.button("Get Recommendations"):
        recommendations = hybrid_recommendation(user_id, movie_title, top_n)

        # Display hybrid recommendations
        if recommendations:
            st.subheader("Recommended Movies:")
            cols = st.columns(3)  # Create 5 columns for tiles
            for idx, (movie, poster_url) in enumerate(recommendations):
                col = cols[idx % 3]  # Cycle through columns
                with col:
                    # Center align image and caption with fixed height
                    st.markdown(f'<div class="movie-tile">'
                                f'<img src="{poster_url}" width="120" /><br>'
                                f'<span>{movie}</span>'
                                f'</div>', unsafe_allow_html=True)
        else:
            st.write("No recommendations found. Try a different movie!")

        # Top genres recommendations
        top_genres = get_user_top_genres(user_id)
        if top_genres:
            for genre in top_genres:
                st.subheader(f"Top 10 Movies in {genre} Genre:")
                top_movies = recommend_top_movies_from_genre(genre)

                # Display top movies for the genre
                genre_cols = st.columns(3)  # 5 tiles per row
                for idx, row in top_movies.iterrows():
                    col = genre_cols[idx % 3]  # Cycle through columns
                    with col:
                        # Center align image and caption with fixed height
                        st.markdown(f'<div class="movie-tile">'
                                    f'<img src="{row["poster_url"]}" width="120" /><br>'
                                    f'<span>{row["title"]}</span>'
                                    f'</div>', unsafe_allow_html=True)
except Exception as e:
    st.error(f"Error in Streamlit app: {traceback.format_exc()}")
