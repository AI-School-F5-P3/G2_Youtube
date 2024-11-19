from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import pickle
import numpy as np

class ToxicityClassifier:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        
    def create_model(self, input_dim):
        model = Sequential([
            Dense(256, input_dim=input_dim, kernel_regularizer=l2(0.01)),
            LeakyReLU(alpha=0.1),
            Dropout(0.5),
            Dense(128, kernel_regularizer=l2(0.01)),
            LeakyReLU(alpha=0.1),
            Dropout(0.5),
            Dense(64, kernel_regularizer=l2(0.01)),
            LeakyReLU(alpha=0.1),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test, vectorizer):
        self.vectorizer = vectorizer
        
        self.model = self.create_model(X_train.shape[1])
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        )
        
        history = self.model.fit(
            X_train,
            y_train,
            epochs=100,
            batch_size=64,
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Predicciones
        y_train_pred = (self.model.predict(X_train) > 0.4).astype("int32")  # Ajustar umbral
        y_test_pred = (self.model.predict(X_test) > 0.4).astype("int32")
        
        # Métricas
        train_metrics = self._calculate_metrics(y_train, y_train_pred)
        test_metrics = self._calculate_metrics(y_test, y_test_pred)
        
        # Análisis de overfitting
        overfitting_gaps = {
            metric: train_metrics[metric] - test_metrics[metric]
            for metric in train_metrics.keys()
        }
        
        results = {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'overfitting_gaps': overfitting_gaps,
            'history': history.history
        }
        
        return results
    
    def _calculate_metrics(self, y_true, y_pred):
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }
    
    def predict(self, texts):
        X = self.vectorizer.transform(texts)
        
        probabilities = self.model.predict(X)
        predictions = (probabilities > 0.4).astype(int)  # Ajustar umbral
        
        return predictions, probabilities
    
    def save_model(self, filepath='toxicity_model'):
        self.model.save(f'{filepath}.h5')
        
        with open(f'{filepath}_vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)

    def plot_training_history(self, history):
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['accuracy'], label='Training Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Cargar datos preprocesados
    with open('processed_data.pkl', 'rb') as f:
        data = pickle.load(f)

    classifier = ToxicityClassifier()
    
    # Entrenar y evaluar
    results = classifier.train_and_evaluate(
       data['X_balanced'],
       data['X_test'],
       data['y_balanced'],
       data['y_test'],
       data['vectorizer']
    )
    
    print("\nMétricas de entrenamiento:")
    for metric, value in results['train_metrics'].items():
       print(f"{metric}: {value:.4f}")
    
    print("\nMétricas de prueba:")
    for metric, value in results['test_metrics'].items():
       print(f"{metric}: {value:.4f}")
    
    print("\nAnálisis de overfitting:")
    for metric, value in results['overfitting_gaps'].items():
       print(f"Diferencia en {metric}: {value:.4f}")
    
    # Guardar el modelo
    classifier.save_model('modelo_toxicidad_final')

    # Graficar el historial de entrenamiento
    classifier.plot_training_history(results['history'])