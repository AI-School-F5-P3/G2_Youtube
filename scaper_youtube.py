from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import sqlite3  # Para base de datos SQLite, puedes cambiar a otra si prefieres


class YouTubeMessageScraper:
    def __init__(self):
        self.driver = webdriver.Chrome()

    def get_messages(self, video_url, num_messages=100):
        """Obtiene solo los mensajes de los comentarios"""
        try:
            print("Obteniendo mensajes...")
            self.driver.get(video_url)
            time.sleep(3)

            # Scroll para cargar comentarios
            for _ in range(5):  # Reducido a 5 scrolls
                self.driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
                time.sleep(2)

            # Obtener los mensajes
            messages = []
            comments = self.driver.find_elements(By.CSS_SELECTOR, "#content-text")

            for comment in comments[:num_messages]:
                try:
                    message = comment.text.strip()
                    if message:  # Solo guardar mensajes no vacíos
                        messages.append(message)
                except Exception as e:
                    print(f"Error processing comment. Skipping... {e}")
                    continue
          
            return messages
        except Exception as e:
            print(f"An error occurred: {e}")
            return []

    def close(self):
        self.driver.quit()


def save_to_database(messages, video_id):
    """Guarda los mensajes en una base de datos SQLite"""
    conn = sqlite3.connect('youtube_messages.db')
    c = conn.cursor()
    
    # Crear tabla si no existe
    c.execute('''CREATE TABLE IF NOT EXISTS messages
                 (video_id TEXT, message TEXT)''')
    
    # Insertar mensajes
    for message in messages:
        c.execute('INSERT INTO messages VALUES (?, ?)', (video_id, message))
    
    conn.commit()
    conn.close()

def main():
    # URL del video
    video_url = "https://www.youtube.com/watch?v=_TqMek9evXs"
    video_id = video_url.split('=')[1]  # Obtener ID del video
    
    scraper = YouTubeMessageScraper()
    try:
        # Obtener mensajes
        messages = scraper.get_messages(video_url, num_messages=100)
        
        if messages:
            # Guardar en base de datos
            save_to_database(messages, video_id)
            
            print(f"\nSe obtuvieron {len(messages)} mensajes")
            print("\nPrimeros 5 mensajes:")
            for i, msg in enumerate(messages[:5], 1):
                print(f"{i}. {msg[:100]}...")  # Mostrar solo primeros 100 caracteres
                
            print("\nMensajes guardados en la base de datos 'youtube_messages.db'")
        else:
            print("No se pudieron obtener mensajes")
            
    finally:
        scraper.close()

if __name__ == "__main__":
    main()