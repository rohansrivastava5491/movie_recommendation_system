import streamlit as st
import pandas as pd
from src.data_import import import_data
from src.preprocessing import preprocess_data
from src.model import train_model, get_movie_recommendation

st.title("Movie Recommendation System")

@st.cache_data
def load_data():
    movies, ratings = import_data()
    final_dataset, no_user_voted, no_movies_voted = preprocess_data(ratings)
    knn, csr_data, metrics = train_model(final_dataset)
    return movies, ratings, final_dataset, no_user_voted, no_movies_voted, knn, csr_data, metrics

movies, ratings, final_dataset, no_user_voted, no_movies_voted, knn, csr_data, metrics = load_data()

# Get movie titles for auto-suggestion
movie_titles = movies['title'].tolist()

# Create tabs
tabs = st.tabs(["Data", "Preprocessing", "Visualization", "Model", "Performance Analysis"])

with tabs[0]:
    st.header("Data")
    st.subheader("Movies Data")
    st.write(movies.head())
    st.subheader("Ratings Data")
    st.write(ratings.head())

with tabs[1]:
    st.header("Preprocessing")
    st.write("Final Dataset after Preprocessing")
    st.write(final_dataset.head())

with tabs[2]:
    st.header("Visualization")
    st.subheader("Number of Users Voted per Movie")
    st.bar_chart(no_user_voted)

    st.subheader("Number of Votes per User")
    st.bar_chart(no_movies_voted)

with tabs[3]:
    st.header("Model")
    # Auto-suggest movie input
    movie_titles = movies['title'].tolist()
    movie_name = st.selectbox("Enter a movie name:", movie_titles)

    # Add a button to trigger recommendations
    if st.button("Recommend"):
        if movie_name:
            recommendations = get_movie_recommendation(movie_name, movies, final_dataset, knn, csr_data)
            if isinstance(recommendations, pd.DataFrame):
                st.write("Recommendations:")
                st.dataframe(recommendations)
            else:
                st.write(recommendations)
        else:
            st.write("Please select a movie name.")

with tabs[4]:
    st.header("Performance Analysis")
    st.write("Performance metrics:")
    st.write(metrics)
