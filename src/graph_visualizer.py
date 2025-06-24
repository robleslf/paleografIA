import json
import networkx as nx
import matplotlib.pyplot as plt

# --- Configuración ---
PROCESSED_DATA_PATH = "knowledge_graph/processed_sample.json"
GRAPH_IMAGE_PATH = "knowledge_graph/knowledge_graph_sample.png"

if __name__ == "__main__":
    print("--- Construyendo y Visualizando el Grafo de Conocimiento ---")
    
    try:
        with open(PROCESSED_DATA_PATH, 'r', encoding='utf-8') as f:
            processed_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{PROCESSED_DATA_PATH}'. Ejecuta el script de procesamiento primero.")
        exit()

    G = nx.DiGraph()

    for graph_part in processed_data:
        for entity in graph_part.get("entidades", []):
            if "canonical_id" in entity: # Asegurarse de que el nodo tiene un ID
                G.add_node(
                    entity["canonical_id"], 
                    tipo=entity.get("tipo", "Desconocido"), 
                    original_id=entity.get("original_id", ""),
                    atributos=str(entity.get("atributos", {}))
                )
    
    for graph_part in processed_data:
        for rel in graph_part.get("relaciones", []):
            # Añadir nodos implícitamente si no existen para evitar errores, aunque no tendrán atributos
            if rel.get("sujeto") and rel.get("objeto"):
                G.add_edge(rel["sujeto"], rel["objeto"], label=rel["predicado"])
    
    print(f"-> Grafo construido con {G.number_of_nodes()} nodos y {G.number_of_edges()} relaciones.")

    print("Generando visualización del grafo...")
    plt.figure(figsize=(25, 25))

    pos = nx.spring_layout(G, k=1.5, iterations=50) 
    
    node_colors = []
    color_map = {
        'Persona': 'skyblue',
        'Organización': 'lightgreen',
        'Lugar': 'salmon',
        'Documento': 'khaki',
        'ConceptoClave': 'plum',
        'Fecha': 'lightgrey',
        'Desconocido': 'grey'
    }

    # --- LÍNEA CORREGIDA ---
    for node_data in G.nodes(data=True):
        tipo_de_nodo = node_data[1].get('tipo', 'Desconocido')
        node_colors.append(color_map.get(tipo_de_nodo, 'grey'))
            
    nx.draw_networkx_nodes(G, pos, node_size=2500, node_color=node_colors, alpha=0.9)
    
    # Reducimos un poco el tamaño de las etiquetas para que no se solapen tanto
    labels = {node: f"{data.get('original_id', node)}" for node, data in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=7, font_family='sans-serif')
    
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=15, edge_color='gray', alpha=0.5, connectionstyle='arc3,rad=0.1')
    
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=6)

    plt.title("Grafo de Conocimiento - Muestra de Actas Históricas")
    plt.axis('off')
    
    print(f"Guardando imagen del grafo en '{GRAPH_IMAGE_PATH}'...")
    plt.savefig(GRAPH_IMAGE_PATH, format="PNG", dpi=300, bbox_inches='tight')
    plt.close()

    print("\n¡Visualización del grafo completada con éxito!")
