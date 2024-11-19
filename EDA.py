def clean_text(text):
    """Limpia el texto convirtiéndolo a minúsculas y elimina puntuación."""
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stop_words(text):
    """Elimina palabras vacías."""
    stop_words = set(["the", "is", "in", "and", "to", "a", "an"])
    return ' '.join([word for word in text.split() if word not in stop_words])

def replace_with_synonyms(text, synonym_dict):
    """Reemplaza palabras en el texto con sinónimos del diccionario."""
    words = text.split()
    new_words = []
    for word in words:
        word_cleaned = clean_text(word)
        if word_cleaned in synonym_dict:
            new_word = random.choice(synonym_dict[word_cleaned])
            new_words.append(new_word)
        else:
            new_words.append(word)
    return ' '.join(new_words)

def replace_expressions(text, expression_dict):
    """Reemplaza expresiones en el texto con equivalentes ofensivos."""
    for expression, replacements in expression_dict.items():
        pattern = re.compile(re.escape(expression), re.IGNORECASE)
        if pattern.search(text):
            replacement = random.choice(replacements)
            text = pattern.sub(replacement, text)
    return text

def preprocess_text(text):
    """Aplica todas las técnicas de preprocesamiento."""
    text = clean_text(text)
    text = remove_stop_words(text)
    return text

def main():
    # Cargar datos desde un archivo CSV
    df = pd.read_csv('data/toxic_english.csv')
    
    # Mantener solo las columnas 'Text' e 'IsToxic'
    df = df[['Text', 'IsToxic']]
    
    # Eliminar duplicados y valores nulos
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    
    # Aumentar los datos
    df['augmented_text'] = df['Text'].apply(lambda x: replace_with_synonyms(x, synonym_dict))
    df['augmented_text'] = df['augmented_text'].apply(lambda x: replace_expressions(x, expression_dict))
    
    # Preprocesar textos
    df['cleaned_text'] = df['Text'].apply(preprocess_text)
    df['augmented_cleaned_text'] = df['augmented_text'].apply(preprocess_text)
    
    # Combinar textos originales y aumentados
    X = pd.concat([df['cleaned_text'], df['augmented_cleaned_text']])
    y = pd.concat([df['IsToxic'], df['IsToxic']])  # Duplicamos las etiquetas
    
    # Vectorización TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    X_vectorized = vectorizer.fit_transform(X)
    
    # Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)
    
    # Guardar los datos procesados en un archivo .pkl
    processed_data = {
        'X_balanced': X_train,
        'X_test': X_test,
        'y_balanced': y_train,
        'y_test': y_test,
        'vectorizer': vectorizer
    }
    
    with open('processed_data.pkl', 'wb') as f:
        pickle.dump(processed_data, f)
    
    print("Datos procesados y guardados en 'processed_data.pkl'")

if __name__ == "__main__":
    main()