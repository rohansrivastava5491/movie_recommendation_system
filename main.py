from src.data_import import import_data
from src.preprocessing import preprocess_data
from src.visualization import visualize_votes
from src.model import train_model, get_movie_recommendation

def main():
    # Import data
    movies, ratings = import_data()

    # Preprocess data
    final_dataset, no_user_voted, no_movies_voted = preprocess_data(ratings)
    
    # Optionally save processed data
    final_dataset.to_csv("data/processed/final_dataset.csv", index=False)

    # Visualize data
    visualize_votes(no_user_voted, no_movies_voted)

    # Train model
    knn, csr_data = train_model(final_dataset)

    # Get recommendations
    movie_name = 'Iron Man'
    recommendations = get_movie_recommendation(movie_name, movies, final_dataset, knn, csr_data)
    print(recommendations)

if __name__ == "__main__":
   main()

   



