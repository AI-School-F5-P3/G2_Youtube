from fetch_comments import fetch_comments
from predict_hate import classify_hate
from database import create_database, save_prediction_to_db

# load the API key
API_KEY = "AIzaSyCel7F1O8f4wbBTSehmS3p1jCLQEbfvpqE"
VIDEO_URL = "https://www.youtube.com/watch?v=_TqMek9evXs&ab_channel=NowThisImpact"


def main():
    # create the database
    create_database()

    # fetch the comments
    print("Fetching comments...")
    comments = fetch_comments(VIDEO_URL, API_KEY)
    print(f"Fetched {len(comments)} comments.")

    # classify the comments
    print("Classifying comments...")
    predictions = classify_hate(comments)
    print(f"Classified {len(predictions)} comments.")

    # save the predictions to the database
    print("Saving predictions to the database...")
    save_prediction_to_db(comments, predictions)
    print("Predictions saved to the database.")


if __name__ == "__main__":
    main()
