import pandas as pd
import numpy as np
import unidecode
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import RandomOverSampler
import json
from scipy import sparse

class ToxicCommentsPreprocessor:
    def __init__(self, input_file):
        """
        Inicializa el preprocesador
        
        Args:
            input_file (str): Ruta al archivo CSV original
        """
        self.input_file = input_file
        self.nlp = spacy.load("en_core_web_sm")
        
    def load_data(self):
        """Carga el dataset original"""
        print("Cargando dataset original...")
        df = pd.read_csv(self.input_file, encoding='utf-8')
        
        # Eliminar columna 'CommentId' si existe y filas con valores nulos
        if 'CommentId' in df.columns:
            df = df.drop('CommentId', axis=1)
        
        # Aquí puedes eliminar la columna que desees, por ejemplo 'IsNationalist'
        if 'IsNationalist' in df.columns:
            df = df.drop('IsNationalist', axis=1)
        
        df = df.dropna().reset_index(drop=True)
        
        return df
    
    def preprocess_text(self, text):
        """Preprocesa un texto individual"""
        # Limpieza básica
        text = str(text).lower().strip()
        text = re.sub(r'[^\w\s]', '', text)
        text = unidecode.unidecode(text)
        
        # Lematización
        doc = self.nlp(text)
        lemmas = [token.lemma_.lower() for token in doc 
                 if not token.is_punct and not token.is_space]
        
        return ' '.join(lemmas)
    
    def generate_features(self, texts):
        """Genera características TF-IDF"""
        vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=2,
            max_df=0.95,
            ngram_range=(1, 3)
        )
        
        return vectorizer, vectorizer.fit_transform(texts)
    
    def balance_classes(self, df):
        """
        Balancea las clases para cada columna de etiquetas directamente en el DataFrame
        
        Args:
            df: DataFrame con las características y las etiquetas
        
        Returns:
            df_balanced: DataFrame balanceado
        """
        print("Balanceando clases...")
        
        # Identificar columnas de características y etiquetas
        feature_columns = [col for col in df.columns if col not in ['Text', 'processed_text'] and col not in df.select_dtypes(include=['bool', 'int64']).columns]
        label_columns = df.select_dtypes(include=['bool', 'int64']).columns
        
        # Crear un DataFrame balanceado
        df_balanced = df[feature_columns].copy()
        
        # Balancear cada etiqueta por separado
        ros = RandomOverSampler(random_state=42)
        for label in label_columns:
            X_temp = df[feature_columns]
            y_temp = df[label]
            X_resampled, y_resampled = ros.fit_resample(X_temp, y_temp)
            
            # Actualizar el DataFrame balanceado
            df_balanced = pd.DataFrame(X_resampled, columns=feature_columns)
            df_balanced[label] = y_resampled
        
        return df_balanced
    
    def process_and_save(self, output_dir='data/processed/'):
        """Procesa el dataset completo, balancea las clases y guarda los resultados"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Cargar datos
        df = self.load_data()
        
        # Preprocesar textos
        print("Preprocesando textos...")
        df['processed_text'] = df['Text'].apply(self.preprocess_text)
        
        # Generar características TF-IDF
        print("Generando características TF-IDF...")
        vectorizer, tfidf_matrix = self.generate_features(df['processed_text'])
        
        # Convertir las columnas booleanas a int
        label_columns = df.select_dtypes(include=['bool']).columns
        df[label_columns] = df[label_columns].astype(int)
        
        # Crear un DataFrame con las características TF-IDF y las etiquetas
        df_features = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
        df_combined = pd.concat([df_features, df[label_columns]], axis=1)
        
        # Balancear clases
        df_balanced = self.balance_classes(df_combined)
        
        # Guardar los diferentes componentes
        print("\nGuardando archivos procesados...")
        
        # 1. DataFrame con datos balanceados
        df_balanced.to_csv(f'{output_dir}balanced_data.csv', index=False)
        
        # 2. Matriz TF-IDF balanceada en formato sparse
        X_resampled = sparse.csr_matrix(df_balanced.drop(columns=label_columns))
        sparse.save_npz(f'{output_dir}balanced_tfidf_matrix.npz', X_resampled)
        
        # 3. Vocabulario del vectorizador
        vocabulary_dict = {str(k): int(v) for k, v in vectorizer.vocabulary_.items()}
        feature_names = vectorizer.get_feature_names_out().tolist()
        
        with open(f'{output_dir}vectorizer_vocab.json', 'w') as f:
            json.dump({
                'vocabulary': vocabulary_dict,
                'feature_names': feature_names
            }, f)
        
        print(f"Archivos guardados en {output_dir}:")
        print("- balanced_data.csv")
        print("- balanced_tfidf_matrix.npz")
        print("- vectorizer_vocab.json")
        
        return df_balanced, X_resampled

# Uso de la clase
preprocessor = ToxicCommentsPreprocessor('data/Hoja de cálculo sin título - youtoxic_english_1000_.csv')
df_balanced, X_resampled = preprocessor.process_and_save()