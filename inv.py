import os
import sys

# --- Importaciones de LangChain y otras librerías ---
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- CONFIGURACIÓN GLOBAL ---
# Ruta a la carpeta con los documentos. Asegúrate de que esta carpeta exista.
DATA_PATH = 'documentos/'
# Ruta donde se guardará la base de datos de búsqueda.
DB_FAISS_PATH = 'vectorstore/db_faiss'
# Modelo para convertir texto a vectores numéricos (ejecutado en CPU).
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
# Modelo de lenguaje a usar con Ollama (el que descargamos).
OLLAMA_MODEL = 'llama3'

# --- PLANTILLA DE PROMPT ---
# Instrucciones genéricas para el modelo de lenguaje.
prompt_template = """
### INSTRUCCIONES ###
Usa el siguiente CONTEXTO para responder la PREGUNTA al final.
El CONTEXTO es la única fuente de información válida.
Si la respuesta no está en el CONTEXTO, indica que la información no está disponible en los documentos proporcionados.

### CONTEXTO ###
{context}

### PREGUNTA ###
{question}

### RESPUESTA ###
"""

def crear_prompt_personalizado():
    """Crea un objeto PromptTemplate con nuestra plantilla."""
    return PromptTemplate(
        template=prompt_template, input_variables=['context', 'question']
    )

def crear_base_de_datos_vectorial():
    """
    Crea la base de datos de búsqueda a partir de los archivos PDF.
    Esta función se ejecuta solo si la base de datos no existe.
    """
    if not os.path.exists(DATA_PATH) or not os.listdir(DATA_PATH):
        print(f"Error: La carpeta '{DATA_PATH}' no existe o está vacía.")
        print("Por favor, crea la carpeta y añade tus archivos PDF antes de continuar.")
        sys.exit(1)

    print(f"Cargando documentos desde '{DATA_PATH}'...")
    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()
    if not documents:
        print("Error: No se pudieron cargar documentos. Asegúrate de que los archivos son PDF válidos y con texto extraíble.")
        sys.exit(1)
    print(f"-> Se han cargado {len(documents)} páginas en total.")

    print("Procesando y dividiendo el texto...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    print(f"-> Texto dividido en {len(texts)} fragmentos.")

    print("Creando la base de datos de búsqueda (esto puede tardar varios minutos)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={'device': 'cpu'})
    
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)
    print(f"-> Base de datos creada y guardada en '{DB_FAISS_PATH}'.")

def crear_cadena_qa():
    """
    Crea la cadena de procesamiento que conecta la búsqueda con la generación de respuesta.
    """
    print("Cargando la base de datos de búsqueda...")
    if not os.path.exists(DB_FAISS_PATH):
        print(f"Error: La base de datos no se encuentra en '{DB_FAISS_PATH}'.")
        print("Ejecuta el script sin argumentos para crearla primero.")
        sys.exit(1)
    
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

    print("Configurando el modelo de lenguaje local...")
    llm = Ollama(model=OLLAMA_MODEL)
    
    prompt = crear_prompt_personalizado()
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 4}), # Busca los 4 fragmentos más relevantes
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain

# --- Flujo Principal de Ejecución ---
if __name__ == "__main__":
    if not os.path.exists(DB_FAISS_PATH):
        print("--- MODO DE INICIALIZACIÓN ---")
        crear_base_de_datos_vectorial()
        print("\nInicialización completada. Ejecuta el script de nuevo para iniciar el modo de consulta.")
    else:
        print("--- MODO DE CONSULTA ---")
        qa_chain = crear_cadena_qa()

        print("\n\n### Sistema de Consulta de Documentos ###")
        print("Introduce tu consulta. Escribe 'salir' para terminar.")
        
        while True:
            question = input("\nConsulta: ")
            if question.lower() == 'salir':
                break
            if not question.strip():
                continue
            
            print("\nProcesando...")
            result = qa_chain.invoke({"query": question})
            
            print("\n### Respuesta ###")
            print(result["result"].strip())
            
            print("\n### Documentos de Referencia ###")
            fuentes = set()
            for doc in result["source_documents"]:
                nombre_archivo = os.path.basename(doc.metadata.get('source', 'N/A'))
                pagina = doc.metadata.get('page', 'N/A')
                if pagina != 'N/A':
                    # pypdf numera las páginas desde 0, le sumamos 1 para que sea más intuitivo
                    pagina_usuario = pagina + 1
                    fuentes.add(f"- Archivo: '{nombre_archivo}', Página: {pagina_usuario}")

            for fuente in sorted(list(fuentes)):
                print(fuente)
