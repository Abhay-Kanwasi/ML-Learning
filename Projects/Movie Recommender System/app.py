import streamlit as st
import pickle
import pandas as pd
import requests

def fetch_poster(movie_id):
    response = requests.get('https://api.themoviedb.org/3/movie/{}?api_key=21350d405110412bfa89dc3ca6ec9b0d'.format(movie_id))
    data = response.json()
    return "https://image.tmdb.org/t/p/w500/" + data['poster_path']

def recommend(movie):
    """
    This function takes movie as an input and return five similar movie as output
    Step1: Calculate the index of the movie in new_dataframe.
    Step2: Use the index as parameter for similarity matrix.
    Step3: Then sort them so we can get the similar movie at the first. Sorting will be in descending order.
    """
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:6]
    recommend_movies = []
    recommend_movies_posters = []
    for i in movies_list:
        movie_id = movies.iloc[i[0]].movie_id
        recommend_movies.append(movies.iloc[i[0]].title)
        recommend_movies_posters.append(fetch_poster(movie_id))
    return recommend_movies, recommend_movies_posters

movies_dict = pickle.load(open('movies_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)

similarity = pickle.load(open('similarity.pkl', 'rb'))

st.title('Movie Recommender System')

selected_movie_name = st.selectbox(
    'How would you like to be contacted ?', movies['title'].values
)

if st.button('Recommend'):
    names, posters = recommend(selected_movie_name)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(names[0])
        st.image(posters[0])
    with col2:
        st.text(names[1])
        st.image(posters[1])
    with col3:
        st.text(names[2])
        st.image(posters[2])
    with col4:
        st.text(names[3])
        st.image(posters[3])
    with col5:
        st.text(names[4])
        st.image(posters[4])