import os
import json
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

# --- Configuración ---
DB_FAISS_PATH = 'vectorstore/db_faiss'
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
OLLAMA_MODEL = 'llama3'
# Las páginas de muestra que seleccionamos del Tomo XI
PAGES_TO_TEST = [4, 5, 6, 13, 25] 

# --- Prompt para la extracción de conocimiento ---
extraction_prompt_template = """
A partir del siguiente CONTEXTO de un acta histórica, extrae todas las entidades y relaciones relevantes.
Tu objetivo es construir un grafo de conocimiento. Debes seguir el siguiente esquema estrictamente:

ESQUEMA:
1.  **Entidades:** Identifica Personas, Lugares, Fechas, Organizaciones, Documentos y Conceptos Clave.
    -   Persona: Nombre completo. Atributos: "titulo" (ej: Conde), "cargo" (ej: Gobernador).
    -   Lugar: Nombre del lugar. Atributos: "tipo" (ej: Concejo, Villa, Ciudad).
    -   Organización: Nombre de la institución. (ej: Junta General, Consejo de Castilla).
    -   Fecha: La fecha del evento.
    -   Documento: El tipo de documento. (ej: Real Provisión, Carta).
    -   ConceptoClave: El tema principal o asunto. (ej: impuestos, defensa de las costas).
2.  **Relaciones:** Conecta las entidades con predicados claros como 'ASISTIÓ_A', 'FUE_NOMBRADO', 'ORDENÓ', 'RECLAMÓ', 'TRATÓ_SOBRE', 'NOTIFICÓ'.

FORMATO DE SALIDA:
Tu respuesta debe ser únicamente un objeto JSON válido, sin ningún texto o explicación adicional.
La estructura del JSON debe ser:
{{
  "entidades": [
    {{"id": "Nombre de la Entidad", "tipo": "TipoDeEntidad", "atributos": {{"key": "value"}} }}
  ],
  "relaciones": [
    {{"sujeto": "ID_Entidad_Sujeto", "predicado": "VERBO_RELACION", "objeto": "ID_Entidad_Objeto"}}
  ]
}}

EJEMPLO:
CONTEXTO: "En la Junta de 1703, el Conde de Toreno, alférez mayor, pidió una leva de soldados."
SALIDA JSON:
{{
  "entidades": [
    {{"id": "Junta de 1703", "tipo": "Organización", "atributos": {{"fecha": "1703"}} }},
    {{"id": "Conde de Toreno", "tipo": "Persona", "atributos": {{"titulo": "Conde", "cargo": "alférez mayor"}} }},
    {{"id": "Leva de soldados", "tipo": "ConceptoClave", "atributos": {{}} }}
  ],
  "relaciones": [
    {{"sujeto": "Conde de Toreno", "predicado": "PIDIÓ", "objeto": "Leva de soldados"}}
  ]
}}

---
CONTEXTO:
{context}
---
SALIDA JSON:
"""

def get_extraction_chain():
    """Configura la cadena de extracción con el modelo LLM y el prompt."""
    prompt = PromptTemplate(
        template=extraction_prompt_template,
        input_variables=["context"],
    )
    llm = Ollama(model=OLLAMA_MODEL, format="json") 
    return prompt | llm

if __name__ == "__main__":
    print("--- Iniciando el Proceso de Extracción de Conocimiento (Muestra de Páginas) ---")

    # 1. Cargar la base de datos vectorial existente
    print(f"Cargando base de datos desde '{DB_FAISS_PATH}'...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    
    # 2. Obtener solo los documentos de las páginas seleccionadas para la prueba
    all_docs = db.docstore._dict
    docs_to_process = []
    target_filename = 'Actas Históricas. Tomo XI (1700-1704).pdf'

    for doc_id, doc in all_docs.items():
        if os.path.basename(doc.metadata.get('source', '')) == target_filename and (doc.metadata.get('page', -1) + 1) in PAGES_TO_TEST:
            docs_to_process.append(doc)
    
    print(f"-> Se han encontrado {len(docs_to_process)} fragmentos correspondientes a las páginas de muestra.")

    # 3. Configurar la cadena de extracción
    extraction_chain = get_extraction_chain()

    # 4. Procesar la muestra
    extracted_data = []
    response = "" # Inicializamos la variable response
    for i, doc in enumerate(docs_to_process):
        page_num = doc.metadata.get('page', -1) + 1
        print(f"\n--- Procesando fragmento de la página {page_num} ({i+1}/{len(docs_to_process)}) ---")
        
        texto_a_mostrar = doc.page_content.replace('\n', ' ').strip()
        print(f"Texto original: '{texto_a_mostrar[:250]}...'")
        
        try:
            response = extraction_chain.invoke({"context": doc.page_content})
            clean_response = response.strip().replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_response)
            data['metadata'] = {'source_page': page_num, 'source_file': target_filename}
            extracted_data.append(data)
            print("-> Datos extraídos con éxito.")
        except Exception as e:
            print(f"!!!!!! Error al procesar el fragmento de la página {page_num}: {e}")
            print(f"Respuesta recibida del modelo que causó el error: {response}")

    # 5. Guardar los resultados de la muestra para su análisis
    output_path = "knowledge_graph/extracted_sample.json"
    print(f"\n--- Guardando resultados de la muestra en '{output_path}' ---")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(extracted_data, f, indent=4, ensure_ascii=False)
    
    print("\n¡Proceso de muestra completado! Revisa el archivo JSON para validar la calidad de la extracción.")
