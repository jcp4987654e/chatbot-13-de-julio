import streamlit as st
import groq
import json

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Chatbot del Instituto 13 de Julio",
    page_icon="🎓",
    layout="centered"
)

# --- CONSTANTES Y CONFIGURACIÓN INICIAL ---

MODELOS = ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768"]

SYSTEM_PROMPT = """
Eres un asistente virtual experto del "Instituto 13 de Julio" llamado "TecnoBot".
Tu única función es responder preguntas sobre el instituto, basándote EXCLUSIVAMENTE en la información proporcionada en el CONTEXTO RELEVANTE.
NO PUEDES usar conocimiento externo o buscar en la web. Tu única fuente de verdad es el contexto.
Si la pregunta del usuario no se puede responder con el contexto, DEBES decir amablemente: "No tengo información sobre ese tema. Mi conocimiento se limita a los datos del instituto. Te sugiero reformular tu pregunta o contactar a secretaría."
No inventes nada. Sé amable, servicial y preséntate como "TecnoBot" en tu primer saludo.
"""

# --- FUNCIONES PRINCIPALES ---

def local_css(file_name):
    """Carga un archivo CSS local."""
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"Error: No se encontró el archivo de estilos '{file_name}'. Asegúrate de que exista en la misma carpeta que este script.")

def cargar_base_de_conocimiento(ruta_archivo='conocimiento.json'):
    """Carga la base de conocimientos desde el archivo JSON."""
    try:
        with open(ruta_archivo, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Error Crítico: No se encontró el archivo '{ruta_archivo}'.")
        return None
    except json.JSONDecodeError:
        st.error(f"Error Crítico: El archivo '{ruta_archivo}' no tiene un formato JSON válido.")
        return None

def buscar_contexto_relevante(query, base_de_conocimiento):
    """Busca palabras clave en la consulta para encontrar información relevante."""
    if base_de_conocimiento is None:
        return "Error: la base de conocimientos no está disponible."
    query_lower = query.lower()
    contexto_encontrado = ""
    # Busca en la información general
    for topic, data in base_de_conocimiento.items():
        if isinstance(data, dict) and 'keywords' in data:
            for keyword in data.get("keywords", []):
                if keyword in query_lower:
                    contexto_encontrado += f"- {data.get('content', '')}\n"
                    break # Evita añadir la misma info varias veces
    # Busca en el material académico
    if "material_academico" in base_de_conocimiento:
        for year, subjects in base_de_conocimiento["material_academico"].items():
            for subject_name, subject_data in subjects.items():
                 for keyword in subject_data.get("keywords", []):
                    if keyword in query_lower:
                        # Formatea una respuesta bonita para el material académico
                        info = f"**{subject_data.get('content', subject_name.replace('_', ' ').title())}**\n"
                        info += f"Profesor: {subject_data.get('profesor', 'No asignado')}\n"
                        # Añade información de evaluaciones si existe
                        if subject_data.get('evaluaciones'):
                            info += "**Próximas Evaluaciones:**\n"
                            for eval in subject_data['evaluaciones']:
                                info += f"  - Fecha: {eval['fecha']}, Temas: {eval['temas']}\n"
                        # Añade información de temas si existe
                        if subject_data.get('temas'):
                             info += "**Temas y Apuntes:**\n"
                             for tema in subject_data['temas']:
                                 info += f"  - [{tema['nombre']}]({tema['apuntes']})\n"
                        contexto_encontrado += info + "\n"
                        break


    if not contexto_encontrado:
        return base_de_conocimiento.get("info_general", {}).get("content", "No se encontró contexto específico.")
    return contexto_encontrado

def generar_respuesta_modelo(cliente_groq, modelo_seleccionado, historial_chat):
    """Envía la petición a la API de Groq."""
    try:
        respuesta = cliente_groq.chat.completions.create(
            model=modelo_seleccionado,
            messages=historial_chat,
            temperature=0.5,
            max_tokens=1024,
        )
        return respuesta.choices[0].message.content
    except Exception as e:
        st.error(f"Ocurrió un error al contactar la API de Groq: {e}")
        return None

# --- APLICACIÓN PRINCIPAL DE STREAMLIT ---

def main():
    # --- Carga de Estilos CSS y Definición del Logo ---
    local_css("style.css") # ¡Cargamos el archivo externo!
    LOGO_URL = "https://i.imgur.com/gJ5Ym2W.png" # ¡CAMBIA ESTA URL POR LA DE TU LOGO OFICIAL!

    # --- Contenido de la App ---
    with st.sidebar:
        st.markdown(f'<img src="{LOGO_URL}" class="sidebar-logo">', unsafe_allow_html=True)
        st.header("Configuración")
        modelo_seleccionado = st.selectbox(
            "Elige tu modelo de IA:",
            MODELOS,
            index=1,
            help="Llama3-70b es más potente, Llama3-8b es más rápido."
        )
        try:
            cliente_groq = groq.Groq(api_key=st.secrets["GROQ_API_KEY"])
        except Exception:
            st.error("API Key de Groq no configurada.")
            st.stop()
        st.info("Este chatbot recuerda la conversación actual para dar respuestas más coherentes.")

    st.title("🎓 Chatbot del Instituto 13 de Julio")

    base_de_conocimiento = cargar_base_de_conocimiento()
    if base_de_conocimiento is None:
        st.stop()

    if "mensajes" not in st.session_state:
        st.session_state.mensajes = [
            {"role": "assistant", "content": "¡Hola! Soy TecnoBot, el asistente virtual del Instituto 13 de Julio. ¿En qué puedo ayudarte?"}
        ]

    # Envolvemos el área del chat en nuestro div personalizado para un estilo robusto
    st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)
    chat_container = st.container(height=500)
    with chat_container:
        for mensaje in st.session_state.mensajes:
            with st.chat_message(mensaje["role"], avatar="🤖" if mensaje["role"] == "assistant" else "🧑‍💻"):
                st.markdown(mensaje["content"], unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if prompt_usuario := st.chat_input("Escribe tu pregunta aquí..."):
        st.session_state.mensajes.append({"role": "user", "content": prompt_usuario})

        contexto_rag = buscar_contexto_relevante(prompt_usuario, base_de_conocimiento)
        system_prompt_con_contexto = f"{SYSTEM_PROMPT}\n\nCONTEXTO RELEVANTE:\n{contexto_rag}"

        historial_para_api = [{"role": "system", "content": system_prompt_con_contexto}]
        mensajes_relevantes = [msg for msg in st.session_state.mensajes if msg['role'] != 'system']
        historial_para_api.extend(mensajes_relevantes[-10:])

        respuesta_bot = generar_respuesta_modelo(cliente_groq, modelo_seleccionado, historial_para_api)
        if respuesta_bot:
            st.session_state.mensajes.append({"role": "assistant", "content": respuesta_bot})

        st.rerun()

if __name__ == "__main__":
    main()
