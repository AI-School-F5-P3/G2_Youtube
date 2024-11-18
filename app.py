import streamlit as st
import pickle
from tensorflow.keras.models import load_model
import numpy as np
import re
import unicodedata

def clean_text(text):
    """
    Limpia y normaliza el texto
    """
    # Convertir a minúsculas
    text = text.lower()
    
    # Normalizar caracteres unicode (acentos, ñ, etc)
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    
    # Eliminar URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Eliminar emails
    text = re.sub(r'\S+@\S+', '', text)
    
    # Eliminar números
    text = re.sub(r'\d+', '', text)
    
    # Eliminar caracteres especiales y puntuación extra
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Eliminar espacios múltiples
    text = re.sub(r'\s+', ' ', text)
    
    # Eliminar espacios al inicio y final
    text = text.strip()
    
    return text

def load_model_and_vectorizer(model_path='toxicity_model'):
    # Cargar el modelo y el vectorizador
    model = load_model(f'{model_path}.h5')
    with open(f'{model_path}_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

def predict_toxicity(text, model, vectorizer):
    # Limpiar el texto
    cleaned_text = clean_text(text)
    
    # Vectorizar el texto
    X = vectorizer.transform([cleaned_text])
    
    # Obtener predicción
    probability = model.predict(X)[0][0]
    is_toxic = probability > 0.5
    return is_toxic, probability, cleaned_text

# Título de la aplicación
st.title("🔍 Detector de Toxicidad en Texto")
st.write("""
Esta aplicación analiza texto para detectar contenido tóxico.
Ingresa un texto y el modelo evaluará su nivel de toxicidad.
""")

# Inicializar el estado de la sesión para el texto si no existe
if 'text_input' not in st.session_state:
    st.session_state.text_input = ""

# Función para limpiar el texto
def clear_text():
    st.session_state.text_input = ""

# Cargar modelo y vectorizador
try:
    model, vectorizer = load_model_and_vectorizer()
    
    # Área de texto para input usando session_state
    text_input = st.text_area(
        "Ingresa el texto a analizar:",
        height=100,
        placeholder="Escribe o pega aquí el texto...",
        key="text_input"
    )

    # Crear una fila con dos columnas para los botones
    col1, col2 = st.columns(2)
    
    # Botón de análisis
    with col1:
        analyze_button = st.button("Analizar texto")
    
    # Botón de limpieza
    with col2:
        clear_button = st.button("Limpiar texto", on_click=clear_text)

    if analyze_button:
        if text_input.strip():
            with st.spinner('Analizando...'):
                is_toxic, probability, cleaned_text = predict_toxicity(text_input, model, vectorizer)
            
            # Mostrar texto limpio
            st.subheader("Texto procesado:")
            col1, col2 = st.columns(2)
            with col1:
                st.text_area("Texto original:", text_input, disabled=True)
            with col2:
                st.text_area("Texto limpio:", cleaned_text, disabled=True)
            
            # Mostrar resultados con formato
            st.subheader("Resultados del análisis:")
            
            # Crear dos columnas para los resultados
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    label="Probabilidad de toxicidad",
                    value=f"{probability:.1%}"
                )
            
            with col2:
                if is_toxic:
                    st.error("🚨 Texto clasificado como TÓXICO")
                else:
                    st.success("✅ Texto clasificado como NO TÓXICO")
            
            # Barra de progreso para visualizar la probabilidad
            st.progress(float(probability))
            
            # Interpretación del resultado
            st.info("""
            💡 Interpretación:
            - 0-25%: Contenido seguro
            - 25-50%: Posible contenido problemático
            - 50-75%: Contenido probablemente tóxico
            - 75-100%: Contenido altamente tóxico
            """)
        else:
            st.warning("⚠️ Por favor, ingresa algún texto para analizar.")
            
except Exception as e:
    st.error(f"Error al cargar el modelo: {str(e)}")
    st.error("Asegúrate de que los archivos del modelo estén en el directorio correcto.")