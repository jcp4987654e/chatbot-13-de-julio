import streamlit as st
import groq
import json # Importamos la librería para manejar archivos JSON

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Chatbot del Instituto 13 de Julio",
    page_icon="�",
    layout="centered"
)

# --- CONSTANTES Y CONFIGURACIÓN INICIAL ---

MODELOS = ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768"]

# 1. COMPORTAMIENTO (SYSTEM PROMPT)
SYSTEM_PROMPT = """
Eres un asistente virtual experto del "Instituto 13 de Julio".
Tu nombre es "TecnoBot". Eres amable, servicial y extremadamente eficiente.
Tu única función es responder preguntas relacionadas con el instituto.
Basa tus respuestas estrictamente en el CONTEXTO RELEVANTE que se te proporciona.
Si la pregunta del usuario no tiene que ver con el instituto o el contexto provisto,
responde amablemente que no puedes ayudar con ese tema, ya que tu especialidad es el instituto.
No inventes información. Si no sabes la respuesta, di que no tienes esa información y que
sugieres contactar a la secretaría.
Siempre preséntate como "TecnoBot" en tu primer saludo.
"""

# --- FUNCIONES PRINCIPALES ---

def cargar_base_de_conocimiento(ruta_archivo='conocimiento.json'):
    try:
        with open(ruta_archivo, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Error Crítico: No se encontró el archivo '{ruta_archivo}'. Asegúrate de que exista en la misma carpeta que este script.")
        return None
    except json.JSONDecodeError:
        st.error(f"Error Crítico: El archivo '{ruta_archivo}' no tiene un formato JSON válido.")
        return None

def buscar_contexto_relevante(query, base_de_conocimiento):
    if base_de_conocimiento is None:
        return "Error: la base de conocimientos no está disponible."
        
    query_lower = query.lower()
    contexto_encontrado = ""
    for keyword, content in base_de_conocimiento.items():
        if keyword in query_lower:
            contexto_encontrado += f"- {content}\n"
    
    if not contexto_encontrado:
        return base_de_conocimiento.get("instituto", "No se encontró contexto específico.")
    return contexto_encontrado

def generar_respuesta_modelo(cliente_groq, modelo_seleccionado, historial_chat):
    try:
        respuesta = cliente_groq.chat.completions.create(
            model=modelo_seleccionado,
            messages=historial_chat,
            temperature=0.7,
            max_tokens=1024,
        )
        return respuesta.choices[0].message.content
    except Exception as e:
        st.error(f"Ocurrió un error al contactar la API de Groq: {e}")
        return None

# --- APLICACIÓN PRINCIPAL DE STREAMLIT ---

def main():
    # --- ESTILOS CSS PERSONALIZADOS ---
    # Inyectamos CSS para darle un look único a la aplicación, usando los colores de tu config.toml
    st.markdown("""
        <style>
        /* Fondo principal de la App */
        [data-testid="stAppViewContainer"] {
            background-color: #2d2a4c; /* Morado oscuro de tu config */
        }

        /* Título principal */
        h1 {
            color: #a1c9f4; /* Celeste pastel de tu config */
            text-shadow: 2px 2px 4px #000000;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #4f4a7d; /* Morado pastel de tu config */
        }
        
        /* Globos de chat del Asistente (Bot) */
        .st-emotion-cache-1c7y2kd {
            background-color: #4f4a7d; /* Morado pastel para el bot */
            border-radius: 15px;
            padding: 1rem;
        }

        /* Globos de chat del Usuario */
        .st-emotion-cache-4oy321 {
             background-color: #3b3861; /* Un tono intermedio para el usuario */
             border-radius: 15px;
             padding: 1rem;
        }

        /* Texto dentro de los globos de chat */
        .st-emotion-cache-1c7y2kd p, .st-emotion-cache-4oy321 p {
            color: #e6e6fa !important; /* Lavanda claro de tu config */
        }

        /* Input de texto del chat */
        [data-testid="stChatInput"] {
            background-color: #4f4a7d;
            border-radius: 10px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("🎓 Chatbot del Instituto 13 de Julio")
    st.write("Tu asistente virtual para consultas sobre el instituto.")

    base_de_conocimiento = cargar_base_de_conocimiento()
    if base_de_conocimiento is None:
        st.stop()

    with st.sidebar:
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
            st.error("API Key de Groq no configurada. Ve a 'Settings > Secrets' y añade tu clave.")
            st.stop()
        st.info("Este chatbot recuerda la conversación actual para dar respuestas más coherentes.")

    if "mensajes" not in st.session_state:
        st.session_state.mensajes = [
            {"role": "assistant", "content": "¡Hola! Soy TecnoBot, el asistente virtual del Instituto 13 de Julio. ¿En qué puedo ayudarte?"}
        ]

    # Contenedor para el historial de chat
    chat_container = st.container(height=400, border=True)
    with chat_container:
        for mensaje in st.session_state.mensajes:
            avatar = "🧑‍💻" if mensaje["role"] == "user" else "🤖"
            with st.chat_message(mensaje["role"], avatar=avatar):
                st.markdown(mensaje["content"])

    if prompt_usuario := st.chat_input("Escribe tu pregunta aquí...", key="chat_input"):
        st.session_state.mensajes.append({"role": "user", "content": prompt_usuario})
        
        contexto_rag = buscar_contexto_relevante(prompt_usuario, base_de_conocimiento)
        system_prompt_con_contexto = f"{SYSTEM_PROMPT}\n\nUsa el siguiente CONTEXTO RELEVANTE para formular tu respuesta:\n{contexto_rag}"
        
        historial_para_api = [{"role": "system", "content": system_prompt_con_contexto}]
        mensajes_relevantes = [msg for msg in st.session_state.mensajes if msg['role'] != 'system']
        historial_para_api.extend(mensajes_relevantes[-10:])
        
        respuesta_bot = generar_respuesta_modelo(cliente_groq, modelo_seleccionado, historial_para_api)
        if respuesta_bot:
            st.session_state.mensajes.append({"role": "assistant", "content": respuesta_bot})
        
        # Forzar la recarga de la página para mostrar el nuevo mensaje
        st.rerun()

if __name__ == "__main__":
    main()
