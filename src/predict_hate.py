from transformers import pipeline

# load the pipeline
classifier = pipeline("text-classification",
                      model="Hate-speech-CNERG/dehatebert-mono")


def classify_hate(comment: str):
    '''
    Classify a comment as hate speech or not using a pre-trained model NLP.

    Parameters:
        comment (str): The comment to classify.

    Returns:
        list: A list of tuples (etiqueta, confianza)
    '''
    results = classifier(comment)
    prediction = [(result["label"], result["score"]) for result in results]
    return prediction
