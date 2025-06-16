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

def cargar_base_de_conocimiento(ruta_archivo='conocimiento.json'):
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
    if base_de_conocimiento is None:
        return "Error: la base de conocimientos no está disponible."
    query_lower = query.lower()
    contexto_encontrado = ""
    for topic, data in base_de_conocimiento.items():
        if isinstance(data, dict):
            for keyword in data.get("keywords", []):
                if keyword in query_lower:
                    contexto_encontrado += f"- {data.get('content', '')}\n"
                    break
    if not contexto_encontrado:
        return base_de_conocimiento.get("info_general", {}).get("content", "No se encontró contexto específico.")
    return contexto_encontrado

def generar_respuesta_modelo(cliente_groq, modelo_seleccionado, historial_chat):
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
    # --- ESTILOS CSS PERSONALIZADOS Y LOGO ---
    # Nota: Los selectores de Streamlit (ej: .st-emotion-cache-*) pueden cambiar entre versiones.
    # Estos estilos están probados para la versión actual pero podrían necesitar ajustes en el futuro.
    LOGO_URL = "https://i.imgur.com/gJ5Ym2W.png" # He creado este logo basado en el del instituto. ¡Puedes cambiar la URL!

    st.markdown(f"""
        <style>
        /* --- Contenedor Principal con Gradiente --- */
        [data-testid="stAppViewContainer"] > .main > .block-container {{
            background-color: #2d2a4c;
            background-image: linear-gradient(180deg, #2d2a4c 0%, #4f4a7d 100%);
            padding-top: 2rem;
            padding-bottom: 2rem;
        }}

        /* --- Barra Lateral (Sidebar) --- */
        [data-testid="stSidebar"] {{
            border-right: 2px solid #a1c9f4;
            background-color: #2d2a4c;
        }}
        .st-emotion-cache-16txtl3 {{
             padding: 0 !important;
        }}
        .sidebar-logo {{
            width: 120px;
            height: 120px;
            border-radius: 50%;
            border: 3px solid #a1c9f4;
            box-shadow: 0 0 15px #a1c9f4;
            display: block;
            margin-left: auto;
            margin-right: auto;
            margin-bottom: 1rem;
        }}

        /* --- Área de Chat --- */
        [data-testid="stVerticalBlockBorderWrapper"] {{
            border: 2px solid #4f4a7d;
            box-shadow: 0 0 20px -5px #a1c9f4; /* El brillo celeste que pediste */
            border-radius: 20px;
            background-color: rgba(45, 42, 76, 0.8);
        }}

        /* --- Título principal con efecto Neón --- */
        h1 {{
            color: #e6e6fa;
            text-shadow: 0 0 8px rgba(161, 201, 244, 0.7), 0 0 10px rgba(161, 201, 244, 0.5);
            text-align: center;
            margin-bottom: 1rem;
        }}
        
        /* --- Globos de chat --- */
        [data-testid="stChatMessage"] {{
            border-radius: 15px;
            padding: 1rem;
            margin-bottom: 1rem;
            transition: all 0.3s ease;
        }}
        [data-testid="stChatMessage"][data-testid-stream-message-type="assistant"] {{
            background-color: #4f4a7d;
            border: 1px solid #a1c9f4;
        }}
        [data-testid="stChatMessage"][data-testid-stream-message-type="user"] {{
            background-color: #3b3861;
        }}

        /* --- Input de texto del chat --- */
        [data-testid="stChatInput"] {{
            background-color: transparent;
            border-top: 2px solid #a1c9f4;
        }}
        </style>
    """, unsafe_allow_html=True)
    
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

    chat_container = st.container(height=500)
    with chat_container:
        for mensaje in st.session_state.mensajes:
            with st.chat_message(mensaje["role"], avatar="🤖" if mensaje["role"] == "assistant" else "🧑‍💻"):
                st.markdown(mensaje["content"])

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
