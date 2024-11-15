import pandas as pd
import numpy as np
from scipy import sparse
import json
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

class ToxicCommentsTrainer:
    def __init__(self, processed_dir='data/processed/', use_rf=False):
        """
         Inicializa el entrenador
        
         Args:
             processed_dir (str): Directorio con los archivos procesados 
             use_rf (bool): Si True usa RandomForest, si False usa LogisticRegression 
         """
        self.processed_dir = processed_dir 
        self.use_rf = use_rf 
        
        # Verificar que el directorio existe 
        if not os.path.exists(processed_dir): 
            raise FileNotFoundError(f"El directorio {processed_dir} no existe")

    def load_processed_data(self):
        """Carga los datos procesados""" 
        print("Cargando datos procesados...")
        try: 
            # Verificar que todos los archivos necesarios existen 
            required_files = [ 
                'balanced_data.csv', 
                'balanced_tfidf_matrix.npz', 
                'vectorizer_vocab.json' 
            ] 
            for file in required_files: 
                if not os.path.exists(os.path.join(self.processed_dir, file)): 
                    raise FileNotFoundError(f"No se encontró el archivo {file}") 
            
            # Cargar DataFrame balanceado 
            df = pd.read_csv(f'{self.processed_dir}balanced_data.csv') 
            
            # Cargar matriz TF-IDF balanceada 
            tfidf_matrix = sparse.load_npz(f'{self.processed_dir}balanced_tfidf_matrix.npz') 
            
            # Cargar vocabulario del vectorizador 
            with open(f'{self.processed_dir}vectorizer_vocab.json', 'r') as f: 
                vectorizer_data = json.load(f) 
            
            # Verificar que las dimensiones coinciden 
            if tfidf_matrix.shape[0] != len(df): 
                raise ValueError("La matriz TF-IDF y el DataFrame tienen diferentes números de muestras") 
            
            return df, tfidf_matrix, vectorizer_data 
        except Exception as e: 
            print(f"Error al cargar los datos procesados: {str(e)}") 
            raise

    def create_classifier(self):
        """Crea el clasificador según la configuración""" 
        if self.use_rf: 
            return RandomForestClassifier( 
                n_estimators=200, 
                max_depth=20, 
                min_samples_split=5, 
                min_samples_leaf=2, 
                max_features='sqrt', 
                class_weight='balanced_subsample', 
                random_state=42,
                n_jobs=-1) 
        else: 
            return LogisticRegression( 
                C=1.0,
                max_iter=1000,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1)

    def evaluate_fold(self, y_true, y_pred, fold=None):
        """Evalúa las métricas para un fold""" 
        metrics = {  
            'accuracy': accuracy_score(y_true, y_pred),  
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),  
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),  
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)  
        }  
        
        fold_text = f"Fold {fold}: " if fold is not None else ""  
        print(f"\n{fold_text}Métricas:")  
        for metric, value in metrics.items():  
            print(f"{metric}: {value:.4f}")  
        
        # Agregar reporte de clasificación detallado  
        print("\nReporte de clasificación detallado:")  
        print(classification_report(y_true, y_pred, zero_division=0))  
        
        return metrics

    def train_and_evaluate(self, n_splits=5, save_results=True):  
        """Entrena y evalúa el modelo usando validación cruzada estratificada 
    
        Args:  
            n_splits (int): Número de folds para la validación cruzada  
            save_results (bool): Si True, guarda los resultados en un archivo JSON  
        """  
        # Cargar datos  
        df, tfidf_matrix, _ = self.load_processed_data()  
        
        # Identificar columnas de etiquetas  
        label_columns = df.select_dtypes(include=['int64']).columns  
        
        # Filtrar clases con pocos ejemplos  
        min_positive_samples = len(df) * 0.01  # 1% del total  
        columns_to_keep = [col for col in label_columns if df[col].sum() >= min_positive_samples]  

        if not columns_to_keep:  
            raise ValueError("No hay suficientes ejemplos positivos en ninguna clase")  

        print(f"\nClases seleccionadas ({len(columns_to_keep)}): {columns_to_keep}")  

        # Inicializar resultados
        all_metrics = {col: {'accuracy': [], 'train_accuracy': []} for col in columns_to_keep}

        # Crear una lista para almacenar los resultados de cada columna
        results_list = []

        # Entrenar un modelo para cada categoría
        for col in columns_to_keep:
            print(f"\nEntrenando modelo para: {col}")
            print(f"Distribución de clase: Positivos={df[col].sum()}, Negativos={len(df)-df[col].sum()}")
            
            y = df[col]
            
            # Validación cruzada estratificada
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(tfidf_matrix, y), 1):
                print(f"\nProcesando fold {fold}/{n_splits}")
                
                # Dividir datos
                X_train = tfidf_matrix[train_idx]
                X_val = tfidf_matrix[val_idx]
                y_train = y.iloc[train_idx]
                y_val = y.iloc[val_idx]
                
                # Crear y entrenar pipeline
                pipeline = Pipeline([
                    ('scaler', StandardScaler(with_mean=False)),
                    ('classifier', self.create_classifier())
                ])
                
                # Aplicar SMOTE si hay desbalanceo significativo
                if y_train.sum() / len(y_train) < 0.2:
                    print(f"Aplicando SMOTE para balancear clase {col}")
                    smote = SMOTE(random_state=42)
                    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
                    pipeline.fit(X_train_balanced, y_train_balanced)
                else:
                    pipeline.fit(X_train, y_train)
                
                # Evaluar
                y_pred_val = pipeline.predict(X_val)
                y_pred_train = pipeline.predict(X_train)
                
                all_metrics[col]['accuracy'].append(accuracy_score(y_val, y_pred_val))
                all_metrics[col]['train_accuracy'].append(accuracy_score(y_train, y_pred_train))

        # Calcular métricas finales y crear DataFrame con resultados
        for col, metrics in all_metrics.items():
            avg_accuracy = np.mean(metrics['accuracy'])
            avg_train_accuracy = np.mean(metrics['train_accuracy'])
            overfitting = avg_train_accuracy - avg_accuracy
            
            results_list.append({
                'Columna': col,
                'Accuracy': avg_accuracy,
                'Overfitting': overfitting
            })
        
        # Crear DataFrame con los resultados
        results_df = pd.DataFrame(results_list)
        
        # Ordenar resultados por accuracy descendente
        results_df = results_df.sort_values('Accuracy', ascending=False).reset_index(drop=True)
        
        # Mostrar resultados
        print("\nResultados Finales:")
        print(results_df.to_string(index=False))
        
        if save_results:
            # Guardar resultados
            results_file = os.path.join(self.processed_dir, 'training_results.csv')
            results_df.to_csv(results_file, index=False)
            print(f"\nResultados guardados en {results_file}")
        
        return results_df
    
# Crear instancia del entrenador    
trainer=ToxicCommentsTrainer(    
    processed_dir='data/processed/',    
    use_rf=False  # False para LogisticRegression , True para RandomForest    
)

# Entrenar y evaluar modelos    
results=trainer.train_and_evaluate(n_splits=5 ,save_results=True)