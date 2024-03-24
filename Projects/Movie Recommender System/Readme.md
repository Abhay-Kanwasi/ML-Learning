# Movie Recommender System
____
Movie recommendation system is a fancy way to describe a process that tries to predict your preferred items based on your or people similar to you. In layman's terms, we can say that a Recommendation System is a tool designed to predict/filter the items as per the user's behavior.

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
| Data | ----->> | Preprocessing | ----->> | Model | ----->> | Website | ----->> | Deploy |
|------|---------|---------------|---------|-------|---------|---------|---------|--------|

### Data 

For data, we will be using Kaggale you can use this [link!](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata/data)

### Preprocessing

From dataset only take the columns which are necessary or needed for movie recommendation.

Selected column list:
1. genres 
2. id
3. keywords
4. title
5. overview
6. cast
7. crew

<i>First we will check weather is there any column which is empty or null then we will check weather these columns contain duplicate data or not.</i>

Then we can start with preprocessing. First we will prepare our data in form of list(genre, cast, keyword, crew, overview)

<b>Selected column list:</b>
1. genre | genre of all the movies convert into a list
2. cast  | top 4 cast from movies
3. keyword  | all keyword of movies in form of list
4. crew  | only add director of each movie
5. overview | convert string to list format

Remove spaces from the data(In case some data like Abhay Kanwasi and Abhay Kumar will create problem because for machine Abhay and Kanwasi is seperate entity)

<b>Data preprocessing</b><br />
Step1: Convert string object into list format <br />
Step2: Convert overview into list format <br />
Step3: Remove space between all entities <br />

Now add all the output of step1, step2 and step3 and make new column name tags. Then create a new dataframe and put movie_id, title, tags in that.
Now make tags column string instead of list then lowercase all the strings.

<b>Text Vectorization</b></br>
Convert tags into vectors. There are several techniques for that but the technique we are using here is 'Bag of Words' <br />
<i>Note: We don't add stop words(ex: are, and, of, from) in vectorization<i/>

