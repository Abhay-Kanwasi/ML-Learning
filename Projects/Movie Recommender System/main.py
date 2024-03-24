import numpy as np
import pandas as pd
import ast  # convert string to list

def convert_object(objects, counts=None, job=None):
    """
    This function takes an object, retrieves values from it, converts a string(Already a list which is inside a
    double quotes) to a list, and then extracts the names from that list of objects. If no 'counts' parameter is
    provided (indicating how many values to fetch), it retrieves all names. If you specify the number of values to
    fetch, provide the 'counts' argument.

    Note: Please don't give unnecessary parameters to this function
    """
    if not counts:
        if job:
            objects_list = []
            for data_object in ast.literal_eval(objects):
                if data_object['job'] == job:
                    objects_list.append(data_object['name'])
                    break
            return objects_list
        else:
            objects_list = []
            for data_object in ast.literal_eval(objects):
                objects_list.append(data_object['name'])
            return objects_list
    else:
        if job:
            objects_list = []
            count = 0
            for data_object in ast.literal_eval(objects):
                if count != counts:
                    if data_object['job'] == job:
                        objects_list.append(data_object['name'])
                        break
                    count += 1
                else:
                    break
            return objects_list
        else:
            objects_list = []
            count = 0
            for data_object in ast.literal_eval(objects):
                if count != counts:
                    objects_list.append(data_object['name'])
                    count += 1
                else:
                    break
            return objects_list


# SettingWithCopyWarning Resolve
pd.options.mode.chained_assignment = None

# Starting of code
movies = pd.read_csv('tmdb_5000_movies.csv')
movie_credits = pd.read_csv('tmdb_5000_credits.csv')

movies = movies.merge(movie_credits, on='title')
shape = movies.shape

# Now we will only have the column which have necessary for better movie recommendation system.
recommend_criteria = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
# print(recommend_criteria.isnull().sum()) # Check if there is duplicate row

recommend_criteria.dropna(inplace=True)  # In case any row is empty
# print(recommend_criteria.duplicated().sum()) # Check if there is duplicate row

# Data preprocessing

# Step1
recommend_criteria['genres'] = recommend_criteria['genres'].apply(convert_object)
recommend_criteria['keywords'] = recommend_criteria['keywords'].apply(convert_object)
recommend_criteria['cast'] = recommend_criteria['cast'].apply(convert_object, counts=5)
recommend_criteria['crew'] = recommend_criteria['crew'].apply(convert_object, job='Director')

# Step 2
recommend_criteria['overview'] = recommend_criteria['overview'].apply(lambda x: x.split())

# Step 3
recommend_criteria['genres'] = recommend_criteria['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
recommend_criteria['keywords'] = recommend_criteria['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
recommend_criteria['cast'] = recommend_criteria['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
recommend_criteria['crew'] = recommend_criteria['crew'].apply(lambda x: [i.replace(" ", "") for i in x])

# Now make tags column
recommend_criteria['tags'] = recommend_criteria['overview'] + recommend_criteria['genres'] + recommend_criteria[
    'keywords'] + recommend_criteria['cast'] + recommend_criteria['crew']

# In this column now make tags as a string
new_dataframe = recommend_criteria[['movie_id', 'title', 'tags']]
new_dataframe['tags'] = new_dataframe['tags'].apply(lambda x: " ".join(x))

# Lowercase all the strings(Recommended)
new_dataframe['tags'] = new_dataframe['tags'].apply(lambda x: x.lower())
print(new_dataframe['tags'])


