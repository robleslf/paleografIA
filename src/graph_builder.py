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
TARGET_FILENAME_HINT = "Tomo XI" # Pista para encontrar el archivo correcto

# --- Prompt para la extracción de conocimiento v2.0 (MUY ESTRICTO) ---
extraction_prompt_template = """
A partir del siguiente CONTEXTO de un acta histórica, tu misión es actuar como un archivista experto y extraer entidades y relaciones en un formato JSON estricto.

### ESQUEMA ESTRICTO DE TIPOS Y ATRIBUTOS
Solo puedes usar los siguientes tipos de entidad: "Persona", "Lugar", "Organizacion", "Fecha", "Documento", "ConceptoClave".

1.  **Persona:**
    -   Atributos OBLIGATORIOS: `nombre_completo`.
    -   Atributos OPCIONALES: `titulo` (ej: "Conde", "Marqués"), `cargo` (ej: "Gobernador", "Alférez mayor").
2.  **Lugar:**
    -   Atributos OBLIGATORIOS: `nombre`.
    -   Atributos OPCIONALES: `subtipo` (solo puede ser "Ciudad", "Concejo", "Villa", "Parroquia", "Reino").
3.  **Organizacion:**
    -   Atributos OBLIGATORIOS: `nombre` (ej: "Junta General", "Consejo de Castilla").
4.  **Fecha:**
    -   Atributos OBLIGATORIOS: `fecha_texto` (la fecha tal como aparece en el texto).
5.  **Documento:**
    -   Atributos OBLIGATORIOS: `tipo_documento` (ej: "Real Provisión", "Carta", "Memorial").
6.  **ConceptoClave:**
    -   Atributos OBLIGATORIOS: `tema`.

### ESQUEMA DE RELACIONES
Usa verbos de acción claros en mayúsculas como predicado. Ejemplos: "ASISTIÓ_A", "FUE_NOMBRADO", "ORDENÓ", "RECLAMÓ", "TRATÓ_SOBRE".

### FORMATO DE SALIDA
Tu respuesta debe ser únicamente un objeto JSON válido, sin explicaciones. Sigue esta estructura:
{{
  "entidades": [
    {{"id": "Nombre normalizado", "tipo": "TipoDeEntidad", "atributos": {{"key": "value"}} }}
  ],
  "relaciones": [
    {{"sujeto": "ID_Entidad_Sujeto", "predicado": "VERBO_RELACION", "objeto": "ID_Entidad_Objeto"}}
  ]
}}

### INSTRUCCIÓN CRÍTICA
Antes de generar la salida, revisa que cada entidad tiene su 'tipo' y sus atributos obligatorios según el esquema. Si una entidad no encaja, no la incluyas. La consistencia es la máxima prioridad.

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
    
    for doc_id, doc in all_docs.items():
        source_path = doc.metadata.get('source', '')
        page_num = doc.metadata.get('page', -1) + 1
        
        # --- LÍNEA CORREGIDA Y MEJORADA ---
        if TARGET_FILENAME_HINT in source_path and page_num in PAGES_TO_TEST:
            docs_to_process.append(doc)
    
    print(f"-> Se han encontrado {len(docs_to_process)} fragmentos correspondientes a las páginas de muestra.")

    if not docs_to_process:
        print("\nADVERTENCIA: No se encontraron fragmentos para procesar. Verifica la configuración `PAGES_TO_TEST` y `TARGET_FILENAME_HINT`.")
        exit()

    # 3. Configurar la cadena de extracción
    extraction_chain = get_extraction_chain()

    # 4. Procesar la muestra
    extracted_data = []
    response = ""
    for i, doc in enumerate(docs_to_process):
        page_num = doc.metadata.get('page', -1) + 1
        print(f"\n--- Procesando fragmento de la página {page_num} ({i+1}/{len(docs_to_process)}) ---")
        
        texto_a_mostrar = doc.page_content.replace('\n', ' ').strip()
        print(f"Texto original: '{texto_a_mostrar[:250]}...'")
        
        try:
            response = extraction_chain.invoke({"context": doc.page_content})
            clean_response = response.strip().replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_response)
            data['metadata'] = {'source_page': page_num, 'source_file': os.path.basename(doc.metadata.get('source'))}
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
