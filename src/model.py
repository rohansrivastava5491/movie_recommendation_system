from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import pandas as pd
import numpy as np

def precision_at_k(actual, predicted, k):
    actual_set = set(actual)
    predicted_set = set(predicted[:k])
    return len(actual_set & predicted_set) / float(k)

def recall_at_k(actual, predicted, k):
    actual_set = set(actual)
    predicted_set = set(predicted[:k])
    return len(actual_set & predicted_set) / float(len(actual_set))

def evaluate_model(csr_data, knn, k=10):
    precisions = []
    recalls = []

    for i in range(min(100, csr_data.shape[0])):  # limiting to 100 users for evaluation
        actual = csr_data[i].indices  # actual items the user interacted with
        distances, indices = knn.kneighbors(csr_data[i], n_neighbors=k)
        predicted = indices.flatten()

        precisions.append(precision_at_k(actual, predicted, k))
        recalls.append(recall_at_k(actual, predicted, k))

    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    
    return avg_precision, avg_recall

def train_model(final_dataset):
    csr_data = csr_matrix(final_dataset.values)
    final_dataset.reset_index(inplace=True)
    
    knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
    knn.fit(csr_data)
    
    avg_precision, avg_recall = evaluate_model(csr_data, knn, k=10)
    metrics = {'Precision@10': avg_precision, 'Recall@10': avg_recall}
    
    return knn, csr_data, metrics

def get_movie_recommendation(movie_name, movies, final_dataset, knn, csr_data):
    n_movies_to_recommend = 10
    # Use regex=False to prevent the pattern from being interpreted as a regex
    movie_list = movies[movies['title'].str.contains(movie_name, case=False, regex=False)]
    if len(movie_list):
        movie_idx = movie_list.iloc[0]['movieId']
        movie_idx = final_dataset[final_dataset['movieId'] == movie_idx].index[0]
        distances, indices = knn.kneighbors(csr_data[movie_idx], n_neighbors=n_movies_to_recommend + 1)
        rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]
        recommend_frame = []
        for val in rec_movie_indices:
            movie_idx = final_dataset.iloc[val[0]]['movieId']
            idx = movies[movies['movieId'] == movie_idx].index
            recommend_frame.append({'Title': movies.iloc[idx]['title'].values[0], 'Distance': val[1]})
        df = pd.DataFrame(recommend_frame, index=range(1, n_movies_to_recommend + 1))
        return df
    else:
        return "No movies found. Please check your input"
