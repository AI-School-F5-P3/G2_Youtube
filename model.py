import pandas as pd
import numpy as np
import random
import re
import pickle

# Diccionario ampliado de sinónimos para palabras inapropiadas
synonym_dict = {
    "bad": [
        "poor", "unpleasant", "negative", "subpar", "inferior", "detrimental", 
        "harmful", "undesirable", "unfavorable", "awful", "terrible", 
        "atrocious", "lousy", "dismal", "infernal", "disagreeable", 
        "unacceptable", "regrettable", "deplorable", "vile", "offensive", 
        "unsatisfactory", "substandard", "crappy", "horrendous", "dire", 
        "grievous", "wretched", "miserable", "abysmal", "hopeless", 
        "shoddy", "depressing", "gross", "nasty", "abhorrent", "foul"
    ],
    "sexist": [
        "slut", "whore", "bimbo", "wench", "hag", "witch", 
        "hussy", "gold-digger", "spinster", "ballbuster", "dumb blonde", 
        "nag", "bitch", "broad", "chick", "doll", "skank", 
        "feminazi", "sugar daddy", "sugar baby", "man up", "throw like a girl", 
        "stay in the kitchen", "macho", "misogynist", "womanizer", 
        "emasculate", "boyish", "sissy", "girly", "trophy wife", "domestic"
    ],
    "racist": [
        "coon", "spic", "chink", "gook", "nigger", "cracker", 
        "honky", "wetback", "jungle bunny", "redskin", "slant-eye", 
        "beaner", "sand nigger", "gypsy", "terrorist", "illegal", 
        "half-breed", "mongrel", "white trash", "yellow peril", "savage", 
        "ape", "barbarian", "tribal", "ghetto", "thug", "oriental", 
        "foreigner", "exotic", "chocolate face", "banana", "curry muncher", 
        "snowflake"
    ],
    "hate": [
        "dislike", "detest", "loathe", "despise", "abhor", "abominate", 
        "resent", "disdain", "scorn", "revile", "revulsion", "enmity", 
        "animosity", "antipathy", "malice", "hostility", "intolerance", 
        "bitterness", "contempt", "detestation", "disrelish", "disgust", 
        "repugnance", "aversion", "loathing", "odium", "spite", "hatred", 
        "animus", "wrath", "vindictiveness", "alienation", "grudge", 
        "rancor", "irritation", "exasperation", "belligerence"
    ],
    
    "stupid": [
        "foolish", "unwise", "silly", "ignorant", "dim-witted", "brainless", 
        "daft", "dull", "dense", "witless", "clueless", "half-witted", 
        "inane", "mindless", "moronic", "simple-minded", "asinine", 
        "imbecilic", "blockheaded", "slow-witted", "ridiculous", "absurd", 
        "laughable", "nonsensical", "idiotic", "senseless", "naive", 
        "obtuse", "vacuous", "preposterous", "gullible", "childish", 
        "unthinking", "harebrained", "brain-dead", "irrational", "blundering"
    ],
   "homophobic": [
        "fag", "dyke", "tranny", "fairy", "fruit", "queen", 
        "sodomite", "pervert", "deviant", "pansy", "flamer", 
        "butch", "queer", "faggot", "breeder", "closeted", "limp wrist", 
        "homosexual agenda", "lesbo", "nelly", "top or bottom", 
        "straight-acting", "muff diver", "invert", "deviant", "cross-dresser", 
        "drag queen", "gayboy", "carpet muncher", "bent", "pillow biter"
    ],  
    "ugly": [
        "unattractive", "hideous", "unsightly", "grotesque", "repulsive", 
        "disfigured", "plain", "homely", "frightful", "monstrous", 
        "gruesome", "displeasing", "repugnant", "unappealing", 
        "offensive-looking", "ghastly", "deformed", "uncomely", 
        "horrendous", "beastly", "gritty", "uninviting", "coarse", 
        "ungraceful", "clumsy", "unflattering"
    ],
    
    "angry": [
        "mad", "irate", "furious", "annoyed", "enraged", "infuriated", 
        "indignant", "resentful", "irritated", "outraged", "seething", 
        "wrathful", "vexed", "incensed", "fuming", "raging", "livid", 
        "cross", "exasperated", "provoked", "heated", "agitated", 
        "choleric", "upset", "boiling", "irascible", "testy", "snappish", 
        "sulky", "argumentative", "fiery", "belligerent", "displeased"
    ],
    
    "sad": [
        "unhappy", "miserable", "downcast", "depressed", "melancholy", 
        "sorrowful", "heartbroken", "despondent", "grief-stricken", 
        "blue", "gloomy", "forlorn", "dejected", "disheartened", 
        "wistful", "woeful", "downhearted", "pained", "crestfallen", 
        "troubled", "discouraged", "disconsolate", "desolate", 
        "tearful", "heavy-hearted", "doleful", "lugubrious", 
        "morose", "mournful", "sullen", "somber", "weeping", 
        "regretful", "longing"
    ],
    
    "fear": [
        "anxiety", "dread", "apprehension", "panic", "terror", "fright", 
        "horror", "trepidation", "nervousness", "unease", "alarm", 
        "cowardice", "timidity", "angst", "phobia", "consternation", 
        "jitters", "worry", "concern", "intimidation", "foreboding", 
        "paranoia", "disquiet", "distress", "trembling", "shaking", 
        "discomposure", "insecurity", "shock", "agitation", "tension", 
        "fearfulness", "hesitation"
    ]
}

# Diccionario de expresiones ofensivas
expression_dict = {
    "go to hell": ["burn in hell", "rot in hell", "fall to the abyss"],
    "shut up": ["silence yourself", "keep quiet", "zip it"],
    "you are stupid": ["you are dumb", "you are ignorant", "you are a fool"],
    # Agrega más expresiones según sea necesario
}

def add_unique_synonyms(existing_dict, new_dict):
    """Agrega sinónimos de new_dict a existing_dict solo si no están repetidos."""
    for key, synonyms in new_dict.items():
        if key not in existing_dict:
            existing_dict[key] = synonyms
        else:
            existing_synonyms = set(existing_dict[key])
            for synonym in synonyms:
                if synonym not in existing_synonyms:
                    existing_dict[key].append(synonym)
    return existing_dict

def clean_text(text):
    """Limpia el texto convirtiéndolo a minúsculas y elimina puntuación."""
    text = text.lower()  # Convertir a minúsculas
    text = re.sub(r'\W', ' ', text)  # Eliminar caracteres no alfanuméricos
    text = re.sub(r'\s+', ' ', text).strip()  # Eliminar espacios extra
    return text

def remove_stop_words(text):
    """Elimina palabras vacías."""
    stop_words = set(["the", "is", "in", "and", "to", "a", "an"])
    return ' '.join([word for word in text.split() if word not in stop_words])

def lemmatize(text):
    """Simulación simple de lematización usando un diccionario."""
    words = text.split()
    lemmatized_words = []
    
    for word in words:
        for key in synonym_dict:
            if word in synonym_dict[key]:
                lemmatized_words.append(key)
                break
        else:
            lemmatized_words.append(word)  # Si no hay sinónimo, se mantiene la palabra original
    
    return ' '.join(lemmatized_words)

def replace_with_synonyms(text, synonym_dict):
    """Reemplaza palabras en el texto con sinónimos del diccionario."""
    words = text.split()
    new_words = []
    
    for word in words:
        word_cleaned = clean_text(word)  # Limpia la palabra
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
    text = lemmatize(text)
    return text

def main():
    # Cargar datos desde un archivo CSV
    df = pd.read_csv('dataset.csv')  # Reemplaza con la ruta correcta de tu dataset
    
    # Mantener solo las columnas 'Text' e 'IsToxic'
    df = df[['Text', 'IsToxic']]
    
    # Eliminar duplicados
    df.drop_duplicates(inplace=True)

    # Verificar si hay valores nulos y eliminarlos (opcional)
    df.dropna(inplace=True)

    print("Texto original:")
    print(df['Text'].head())

    # Aumentar los datos reemplazando palabras en el texto con sinónimos
    df['augmented_text'] = df['Text'].apply(lambda x: replace_with_synonyms(x, synonym_dict))
    
    # Aumentar los datos reemplazando expresiones ofensivas
    df['augmented_text'] = df['augmented_text'].apply(lambda x: replace_expressions(x, expression_dict))

    print("\nTexto aumentado:")
    print(df[['Text', 'augmented_text']].head())

    # Preprocesar textos
    df['cleaned_text'] = df['Text'].apply(preprocess_text)
    
    print("\nTexto preprocesado:")
    print(df[['Text', 'cleaned_text']].head())

    # Guardar el DataFrame modificado en un nuevo archivo .pkl
    with open('processed_data.pkl', 'wb') as f:
        pickle.dump(df, f)

if __name__ == "__main__":
    main()
