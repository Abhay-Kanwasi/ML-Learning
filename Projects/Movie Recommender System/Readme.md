# Movie Recommender System
A movie recommendation system is a fancy way to describe a process that tries to predict your preferred items based on your or people similar to you. In layman's terms, we can say that a Recommendation System is a tool designed to predict/filter the items according to the user's behavior.

## How does this project work?
First download the file using this [link!](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata/data) and put it in the same directory where the app.py and main.py then create a virtual env outside of this directory using this command `python -m venv <your-venv-name>` after that run the command `pip install -r requirements.txt` run main.py after that you can use the command to runserver of streamlit.

This project contains two main scripts one is main.py and one is app.py.

<b>main.py</b> 

    Data Processing:
    Reads movie data from CSV files.
    Merges relevant columns for recommendation.
    Filters out unnecessary data and handles missing values and duplicates.

    Text Preprocessing:
    Converts strings to lists.
    Extracts relevant information like genres, keywords, cast, and crew.
    Cleans and preprocesses text data (e.g., removing spaces, stemming).

    Vectorization:
    Utilizes CountVectorizer to convert text data into numerical vectors.
    Calculates cosine similarity between vectors to determine movie similarity.

    Recommendation:
    Defines a function to recommend similar movies based on a given movie.
    Calculates similarity scores between movies and returns the top 5 recommendations.

    Serialization:
    Serializes the preprocessed data and similarity matrix for future use.

<b>app.py</b>

    Fetching Movie Data:
    Utilizes the TMDB API to fetch movie posters based on movie IDs.

    Recommendation Function:
    Defines a function to recommend similar movies based on a given movie.
    Calculates similarity scores between movies and returns the top 5 recommendations along with their posters.

    User Interface:
    Uses Streamlit to create a simple user interface.
    Allows users to select a movie from a dropdown menu.
    Upon clicking the "Recommend" button, displays the top 5 recommended movies along with their posters.

    Loading Preprocessed Data:
    Loads preprocessed movie data and similarity matrix using pickle.
    
    Displaying Recommendations:
    Displays recommended movies and their posters in a grid layout.

First `main.py` will run and give you `movies_dict.pkl` and `similarity.pkl` then in `app.py` you can use them. 

Then you will go to terminal and type this command <br/>
`streamlit run app.py`

### Types of Recommender System
____

1. Content based <br />
    It will recommend content based on the similarity of content.
2. Collaborative filtering based <br />
    It recommend content based on user interest.
    Example: Suppose there is UserA and UserB and both have similar interests. Now if UserA likes some movie then it can be suggested to UserB because both users are consuming same type of content.
3. Hybrid <br />
    Content based + Collaborative filtering based

In this project we are using CONTENT BASED RECOMMENDER SYSTEM.


### Project Flow
_____
| Data | ----->> | Preprocessing | ----->> | Model | ----->> | Website |
|------|---------|---------------|---------|-------|---------|---------|

## Data 

For data, we will be using Kaggale you can use this [link!](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata/data)

## Preprocessing

From the dataset only take the columns which are necessary or needed for movie recommendation.

Selected column list:
1. genres 
2. id
3. keywords
4. title
5. overview
6. cast
7. crew

<i>First we will check weather is there any column which is empty or null then we will check weather these columns contain duplicate data or not.</i>

Then we can start with preprocessing. First, we will prepare our data in the form of a list(genre, cast, keyword, crew, overview)

<b>Selected column list:</b>

    1. genre | genre of all the movies convert into a list
    2. cast  | top 4 cast from movies
    3. keyword  | all keywords of movies in the form of a list
    4. crew  | only add director of each movie
    5. overview | convert string to list format

Remove spaces from the data(In case some data like Abhay Kanwasi and Abhay Kumar will create problems because for the machine Abhay and Kanwasi is a separate entity)

## Data preprocessing
    Step1: Convert string object into list format 
    Step2: Convert overview into list format 
    Step3: Remove space between all entities 

Now add all the output of step 1, step 2, and step 3 and make new column name tags. Then create a new data frame and put movie_id, title, and tags in that.
Now make tags column string instead of list then lowercase all the strings.

## Model
<b>Text Vectorization</b></br>
Convert tags into vectors. There are several techniques for that but the technique we are using here is 'Bag of Words' <br />
<i>Note: We don't add stop words(ex: are, and, of, from) in vectorization</i>

Make an object of CountVectorizer. Then vectorize new_dataframe['tags]. Then transform this into an array.
Remove same words from the array(eg. action and actions are same, similarly actor and actors are same)
Now it's in the form of vectors. Now we will calculate the distance between vectors(means distance from each other) We will not use Euclidean distance(calculate points) we will use Cosign distance(calculate the angle between them).

[Note: Much distance means less similarity. Distance inversely proportional to similarity]

#### Make a recommend function that will return the top 5 similar movies

This function takes a movie as an input and returns five similar movies as output


    Step 1: Calculate the index of the movie in new_dataframe. So we can get the exact movie.


    Step 2: Use the index as a parameter for the similarity matrix. Now get all the distances from the movie.


    Step 3: Then sort them so we can get a similar movie at the first. Sorting will be in descending order.

Now for this sorted matrix if we do it like this: `sorted(similarity[movie_index], reverse=True)`. 


We are checking the similarity based on the index and if we sort it it will change the original index and we can't say that which movie is it comparing with. So for this, we will use the 'enumerate' keyword like this: `list(enumerate(similarity[movie_index], reverse=True))` 


It will get the list then we can sort this data like this: `sorted(list(enumerate(similarity[movie_index])), reverse = True)`


But the problem arises that it is performing sorting based on the first value but we want this according to the second value so we will do it like this: `sorted(list(enumerate(similarity[movie_index])), reverse=True, key=lambda x:x[1])`


Then we will slice top 5 like this: `sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:6]`


## Website

For the creation of this page, we use the Streamlit framework. On this page user will search and select the movie and based on that input this app will suggest 5 similar movies to a user.![Screenshot from 2024-03-24 20-41-21](https://github.com/Abhay-Kanwasi/ML-Learning/assets/78997764/63bdb933-57a1-4abe-82a1-a9d11ea7b597)




