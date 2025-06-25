import json
import os
import re
import networkx as nx
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from src.data_processor import normalize_name_to_id

# --- Configuración ---
PROCESSED_DATA_PATH = "knowledge_graph/processed_sample.json"
DB_FAISS_PATH = 'vectorstore/db_faiss'
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
OLLAMA_MODEL = 'llama3'

# --- PLANTILLAS DE PROMPT ---
rag_prompt_template = "..." # (Pega el prompt de la versión anterior)
router_prompt_template = "..." # (Pega el prompt de la versión anterior)
# (Pegando los prompts para evitar errores)
rag_prompt_template = """
Eres un historiador experto. Usa el siguiente CONTEXTO para responder la PREGUNTA.
Si la información no está en el contexto, di que no has encontrado detalles específicos en los fragmentos consultados.
CONTEXTO: {context}
PREGUNTA: {question}
RESPUESTA:
"""
router_prompt_template = """
Eres un experto en análisis de preguntas. Tu tarea es clasificar la pregunta del usuario en una de dos categorías: "graph" o "vector".
-   Usa "graph" para preguntas que buscan hechos específicos, relaciones, conteos o listas sobre entidades conocidas (personas, lugares, organizaciones). Ejemplos: '¿Quién asistió a la Junta de 1703?', '¿Cuántos concejos se mencionan?', '¿Qué relación hay entre el Conde de Toreno y Juan Blasco?', 'Lista los cargos de Felipe Robles López'.
-   Usa "vector" para preguntas abiertas, que piden resúmenes, explicaciones o información contextual amplia. Ejemplos: '¿Cuál era la situación económica en 1702?', 'Resume las disputas sobre el comercio de grano', 'Explica la importancia de la Guerra de Sucesión'.
Basándote en la pregunta del usuario, responde únicamente con la palabra "graph" o "vector".
Pregunta: {question}
Clasificación:
"""

class QueryEngine:
    def __init__(self):
        print("Inicializando el Motor de Consulta de paleografIA (v4.3)...")
        self.graph = self.load_knowledge_graph()
        self.rag_chain = self.setup_rag_chain()
        self.router_chain = self.setup_router_chain()

    def load_knowledge_graph(self):
        print(f"Cargando Grafo de Conocimiento desde '{PROCESSED_DATA_PATH}'...")
        try:
            with open(PROCESSED_DATA_PATH, 'r', encoding='utf-8') as f:
                processed_data = json.load(f)
        except FileNotFoundError:
            print(f"Error: No se encontró '{PROCESSED_DATA_PATH}'.")
            return nx.DiGraph()
        G = nx.DiGraph()
        for graph_part in processed_data:
            for entity in graph_part.get("entidades", []):
                if "canonical_id" in entity and entity["canonical_id"]: G.add_node(entity["canonical_id"], **entity)
            for rel in graph_part.get("relaciones", []):
                if rel.get("sujeto") and rel.get("objeto"): G.add_edge(rel["sujeto"], rel["objeto"], label=rel["predicado"])
        print(f"-> Grafo cargado con {G.number_of_nodes()} nodos y {G.number_of_edges()} relaciones.")
        return G

    def setup_router_chain(self):
        prompt = PromptTemplate(template=router_prompt_template, input_variables=["question"])
        llm = Ollama(model=OLLAMA_MODEL)
        return prompt | llm

    def setup_rag_chain(self):
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={'device': 'cpu'})
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        llm = Ollama(model=OLLAMA_MODEL)
        prompt = PromptTemplate(template=rag_prompt_template, input_variables=["context", "question"])
        return RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=db.as_retriever(search_kwargs={'k': 7}), return_source_documents=True, chain_type_kwargs={'prompt': prompt})

    def query_graph(self, question: str) -> str:
        print("-> (Decisión: Usando el Grafo de Conocimiento)")
        q = question.lower().strip().replace("¿", "").replace("?", "")
        
        match_count = re.search(r'cuánt[oa]s ([\w\s_]+?)(?: hay| existen| aparecen| diferentes|$)', q)
        match_list = re.search(r'(?:lista|muéstrame)(?: todos los| todas las)? ([\w\s_]+)', q)

        if match_count or match_list:
            match = match_count or match_list
            tipo_entidad_plural = match.group(1).strip()
            tipo_entidad_singular = re.sub(r'es$', '', tipo_entidad_plural).rstrip('s')
            found_entities = set()
            for _, data in self.graph.nodes(data=True):
                tipo_nodo = (data.get('tipo') or "").lower()
                subtipo_nodo = (data.get('atributos', {}).get('subtipo') or "").lower()
                if tipo_entidad_singular in tipo_nodo or tipo_entidad_singular in subtipo_nodo:
                    found_entities.add(data.get('original_id', ''))
            if match_count:
                return f"Según el grafo de la muestra, he encontrado {len(found_entities)} entidades de tipo '{tipo_entidad_singular}'."
            else:
                if found_entities:
                    respuesta = f"Lista de entidades de tipo '{tipo_entidad_singular}' encontradas en la muestra:\n"
                    respuesta += "\n".join(f"- {name}" for name in sorted(list(found_entities)) if name)
                    return respuesta
                else: return f"No encontré entidades de tipo '{tipo_entidad_singular}' en el grafo de la muestra."
        
        if 'sobre' in q:
            try:
                entity_name_raw = q.split('sobre ')[1].strip()
                entity_id_norm_buscado = normalize_name_to_id(entity_name_raw)

                matching_nodes = set()
                for node, data in self.graph.nodes(data=True):
                    if node == entity_id_norm_buscado or entity_name_raw.lower() in (data.get('original_id', '') or "").lower():
                        matching_nodes.add(node)
                
                if not matching_nodes: return f"No se encontró ninguna entidad que coincida con '{entity_name_raw}'."
                
                respuesta = f"Información agregada sobre '{entity_name_raw}':\n"
                atributos_agregados, rel_salientes, rel_entrantes = {}, set(), set()
                
                for node_id in matching_nodes:
                    data = self.graph.nodes[node_id]
                    for key, value in data.get('atributos', {}).items():
                        if key not in atributos_agregados: atributos_agregados[key] = set()
                        if isinstance(value, str) and value: atributos_agregados[key].add(value)
                    for _, target, edge_data in self.graph.out_edges(node_id, data=True):
                        if target in self.graph.nodes:
                            target_name = self.graph.nodes[target].get('original_id', target)
                            if target_name.lower() != entity_name_raw.lower():
                                rel_salientes.add(f"- {edge_data.get('label', 'RELACIONADO_CON')} -> {target_name}")
                    for source, _, edge_data in self.graph.in_edges(node_id, data=True):
                         if source in self.graph.nodes:
                            source_name = self.graph.nodes[source].get('original_id', source)
                            if source_name.lower() != entity_name_raw.lower():
                                rel_entrantes.add(f"- {source_name} -> {edge_data.get('label', 'RELACIONADO_CON')}")
                
                if atributos_agregados:
                    respuesta += "\n  Atributos/Cargos conocidos:\n"
                    for key, values in atributos_agregados.items(): respuesta += f"    - {key.capitalize()}: {', '.join(values)}\n"
                if rel_salientes:
                    respuesta += "\n  Actuó como sujeto en las siguientes relaciones:\n" + "\n".join(sorted(list(rel_salientes))) + "\n"
                if rel_entrantes:
                    respuesta += "\n  Apareció como objeto en las siguientes relaciones:\n" + "\n".join(sorted(list(rel_entrantes))) + "\n"
                return respuesta.strip()
            except IndexError: pass
        
        return "No pude interpretar tu pregunta para el grafo. Intenta con 'cuántos...', 'lista...' o '...sobre [entidad]'."

    def query(self, question: str):
        if not self.graph.nodes: return "Error: El Grafo de Conocimiento no está cargado o está vacío."
        route = self.router_chain.invoke({"question": question}).strip().lower()
        if "graph" in route: return self.query_graph(question)
        else:
            print("-> (Decisión: Usando la Búsqueda Vectorial RAG)")
            result = self.rag_chain.invoke({"query": question})
            fuentes = "\n\n### Documentos de Referencia ###\n"
            fuentes_set = set()
            for doc in result["source_documents"]:
                nombre_archivo = os.path.basename(doc.metadata.get('source', 'N/A'))
                pagina = doc.metadata.get('page', 'N/A')
                if pagina != 'N/A': pagina_usuario = pagina + 1; fuentes_set.add(f"- Archivo: '{nombre_archivo}', Página: {pagina_usuario}")
            return result["result"] + fuentes + "\n".join(sorted(list(fuentes_set)))

if __name__ == "__main__":
    engine = QueryEngine()
    print("\n\n### paleografIA: Motor de Consulta Híbrido (v4.3) ###")
    print("Introduce tu consulta. Escribe 'salir' para terminar.")
    while True:
        user_question = input("\n> Consulta: ")
        if user_question.lower() == 'salir': break
        if not user_question.strip(): continue
        print("\nProcesando...")
        answer = engine.query(user_question)
        print("\n--- Respuesta ---")
        print(answer)
