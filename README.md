# ML-Learning
____
## Learning

### Statics
## Projects

### Movie Recommender System
____

The movie recommendation system operates through two distinct yet interconnected stages: data preparation and user interaction. In the initial phase, exemplified by the first code snippet, the system undertakes crucial backend processes. It meticulously preprocesses the movie dataset, distilling pertinent information such as genres, keywords, cast, and crew. Through techniques like CountVectorizer, textual data undergoes transformation into numerical vectors, enabling the calculation of cosine similarity metrics between different movies. This comprehensive preprocessing culminates in the creation of a similarity matrix, encapsulating the relationships between films based on their features. Concurrently, the system serializes this processed data for efficient storage and retrieval in subsequent operations. In the subsequent phase, represented by the second code snippet, the system engages with users through an intuitive interface crafted using Streamlit. Users navigate through a streamlined interface, selecting a movie of interest from a dropdown menu. Upon triggering the recommendation process, the system leverages the precomputed similarity matrix to suggest the top 5 movies akin to the user's selection. To enrich the user experience, the system dynamically fetches movie posters via the TMDB API, enhancing the visual appeal of the recommendations. This interactive interface fosters personalized movie recommendations, empowering users to explore new cinematic experiences tailored to their preferences seamlessly.


![Screenshot from 2024-03-24 20-40-56](https://github.com/Abhay-Kanwasi/ML-Learning/assets/78997764/8ff6ade0-a934-4591-b16e-37c3062f15e6)
