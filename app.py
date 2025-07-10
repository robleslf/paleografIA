import streamlit as st
from src.query_engine import QueryEngine # Importamos nuestro motor

# --- Configuración de la Página de Streamlit ---
st.set_page_config(
    page_title="paleografIA",
    page_icon="📜",
    layout="wide", # Usamos un layout ancho para más espacio
    initial_sidebar_state="expanded"
)

# --- Estado de la Sesión ---
# Streamlit vuelve a ejecutar el script con cada interacción.
# Usamos st.session_state para guardar objetos y que no se recarguen constantemente.
if 'engine' not in st.session_state:
    # Inicializamos el motor de consulta SOLO la primera vez que se carga la app
    st.session_state.engine = QueryEngine()

if 'messages' not in st.session_state:
    # Inicializamos el historial de la conversación
    st.session_state.messages = []

# --- Diseño de la Interfaz ---

# 1. Barra Lateral (Sidebar)
with st.sidebar:
    st.title("📜 paleografIA")
    st.markdown("""
    **Asistente de Investigación Inteligente para las Actas Históricas de la Junta General del Principado de Asturias (1594-1718).**
    
    Realice consultas en lenguaje natural sobre el corpus documental. El sistema decidirá si usar el grafo de conocimiento para preguntas específicas o la búsqueda de texto para preguntas abiertas.
    """)
    st.markdown("---")
    
    # Mostramos las estadísticas del grafo
    st.subheader("Estadísticas del Corpus (Muestra)")
    if st.session_state.engine.graph:
        num_nodos = st.session_state.engine.graph.number_of_nodes()
        num_relaciones = st.session_state.engine.graph.number_of_edges()
        st.metric(label="Entidades Reconocidas (Nodos)", value=num_nodos)
        st.metric(label="Relaciones Detectadas (Aristas)", value=num_relaciones)
    else:
        st.error("No se pudo cargar el grafo de conocimiento.")

    st.markdown("---")
    st.info("Proyecto en fase de prototipo. Desarrollado por [Tu Nombre/Proyecto].")

# 2. Área Principal (Chat)
st.header("Sistema de Consulta Interactivo")

# Mostrar mensajes del historial
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Capturar la nueva pregunta del usuario
if prompt := st.chat_input("Realice su consulta aquí..."):
    # Añadir la pregunta del usuario al historial y mostrarla
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generar y mostrar la respuesta de la IA
    with st.chat_message("assistant"):
        with st.spinner("Procesando consulta..."):
            # ¡Aquí ocurre la magia! Llamamos a nuestro motor.
            response = st.session_state.engine.query(prompt)
            st.markdown(response)
    
    # Añadir la respuesta de la IA al historial
    st.session_state.messages.append({"role": "assistant", "content": response})
