import sys
import yaml
from pydantic import BaseModel, Field
from typing import Dict, Any

# --- Pydantic Models for Validation ---
class ModelSettings(BaseModel):
    embedding_id: str
    reranker_id: str
    ollama_llm: str
    query_expander_id: str
    condenser_model_id: str
    ollama_options: Dict[str, Any]

class DatabaseSettings(BaseModel):
    persist_path: str
    collection_name: str
    force_rebuild: bool

class ProcessingSettings(BaseModel):
    initial_split_level: int = Field(..., gt=0)
    max_chars_per_chunk: int = Field(..., gt=0)
    min_chars_per_chunk: int = Field(..., gt=0)

class PipelineSettings(BaseModel):
    enable_conversation_memory: bool
    use_reranker: bool
    enable_query_expansion: bool
    query_expansion_char_threshold: int
    retrieval_top_k: int
    default_retrieval_top_k: int
    min_chunks_to_llm: int
    max_chunks_to_llm: int
    min_absolute_score_threshold: float
    min_chunks_for_gap_detection: int
    gap_detection_factor: float
    small_epsilon: float

class SystemSettings(BaseModel):
    low_vram_mode: bool

class AppSettings(BaseModel):
    models: ModelSettings
    database: DatabaseSettings
    processing: ProcessingSettings
    pipeline: PipelineSettings
    system: SystemSettings

def load_settings(config_path: str = "helper/config.yaml") -> AppSettings:
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
        if not config_data:
            raise ValueError("Konfigurationsdatei ist leer oder konnte nicht geparst werden.")
        return AppSettings(**config_data)
    except FileNotFoundError:
        print(f"FEHLER: Konfigurationsdatei nicht gefunden unter '{config_path}'")
        raise
    except Exception as e:
        print(f"FEHLER: Ein Fehler ist beim Laden oder Validieren der Konfiguration aufgetreten: {e}")
        raise

try:
    settings = load_settings()
    print("Konfiguration erfolgreich geladen und validiert.")
except Exception:
    sys.exit(1)