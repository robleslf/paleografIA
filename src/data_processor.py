import json
import os
import re

# --- Configuración ---
# Ruta del archivo JSON "en bruto" extraído por la IA
RAW_EXTRACTION_PATH = "knowledge_graph/extracted_sample.json"
# Ruta donde guardaremos el archivo JSON "limpio y procesado"
PROCESSED_DATA_PATH = "knowledge_graph/processed_sample.json"

def normalize_name_to_id(name: str) -> str:
    """
    Convierte un nombre de entidad en un ID canónico y consistente.
    Ej: "El Señor don Juan Blasco de Orozco" -> "juan_blasco_de_orozco"
    """
    if not isinstance(name, str):
        return ""
    
    # Convertir a minúsculas
    norm_name = name.lower()
    # Quitar títulos y otras palabras comunes
    titles_to_remove = ["don ", "doña ", "el señor ", "señor ", "fray "]
    for title in titles_to_remove:
        norm_name = norm_name.replace(title, "")
    
    # Quitar caracteres especiales, excepto letras, números y espacios
    norm_name = re.sub(r'[^a-z0-9\sñáéíóúü]', '', norm_name)
    # Reemplazar espacios múltiples con uno solo y quitar espacios al inicio/final
    norm_name = re.sub(r'\s+', ' ', norm_name).strip()
    # Reemplazar espacios con guiones bajos para crear un ID válido
    norm_name = norm_name.replace(' ', '_')
    
    return norm_name

if __name__ == "__main__":
    print(f"--- Iniciando el Procesamiento y Limpieza de Datos ---")
    print(f"Leyendo datos en bruto desde: {RAW_EXTRACTION_PATH}")

    # 1. Cargar el archivo JSON en bruto
    try:
        with open(RAW_EXTRACTION_PATH, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{RAW_EXTRACTION_PATH}'. Asegúrate de que el script de extracción se ha ejecutado primero.")
        exit()
    except json.JSONDecodeError:
        print(f"Error: El archivo '{RAW_EXTRACTION_PATH}' no es un JSON válido.")
        exit()

    processed_knowledge = []
    
    print(f"Procesando {len(raw_data)} fragmentos extraídos...")

    # 2. Iterar sobre cada fragmento extraído para limpiarlo
    for raw_graph in raw_data:
        clean_graph = {
            "entidades": [],
            "relaciones": [],
            "metadata": raw_graph.get("metadata", {})
        }
        
        # Mapa para rastrear los IDs originales a los nuevos IDs canónicos
        id_mapping = {}

        # 3. Limpiar y normalizar las entidades
        for entity in raw_graph.get("entidades", []):
            original_id = entity.get("id")
            if not original_id:
                continue # Omitir entidades sin ID

            canonical_id = normalize_name_to_id(original_id)
            if not canonical_id:
                continue # Omitir si el ID normalizado queda vacío

            id_mapping[original_id] = canonical_id
            
            clean_entity = {
                "canonical_id": canonical_id,
                "original_id": original_id,
                "tipo": entity.get("tipo", "Desconocido"),
                # Corregimos el posible typo "atributes"
                "atributos": entity.get("atributos") or entity.get("atributes", {})
            }
            clean_graph["entidades"].append(clean_entity)
        
        # 4. Limpiar y normalizar las relaciones
        for relation in raw_graph.get("relaciones", []):
            sujeto = relation.get("sujeto")
            predicado = relation.get("predicado")
            objeto = relation.get("objeto")

            # Omitir relaciones incompletas o de baja calidad
            if not all([sujeto, predicado, objeto]):
                continue

            # Usar los IDs canónicos en las relaciones
            sujeto_id = id_mapping.get(sujeto, normalize_name_to_id(sujeto))
            objeto_id = id_mapping.get(objeto, normalize_name_to_id(objeto))
            
            clean_relation = {
                "sujeto": sujeto_id,
                "predicado": predicado.upper().replace(" ", "_"), # Estandarizar el predicado
                "objeto": objeto_id
            }
            clean_graph["relaciones"].append(clean_relation)
            
        processed_knowledge.append(clean_graph)

    # 5. Guardar los datos procesados
    print(f"\n--- Guardando datos procesados en '{PROCESSED_DATA_PATH}' ---")
    with open(PROCESSED_DATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(processed_knowledge, f, indent=4, ensure_ascii=False)
        
    print(f"¡Limpieza completada! Se han procesado y guardado {len(processed_knowledge)} grafos de conocimiento.")
