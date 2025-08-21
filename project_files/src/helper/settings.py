# src/settings.py

import yaml
from pydantic import BaseModel, Field
from typing import Dict, Any

# --- Pydantic-Modelle zur Validierung ---
# Diese Modelle definieren die erwartete Struktur und die Datentypen der config.yaml-Datei.
# Falls das YAML-File ein Feld vermisst oder einen falschen Typ hat, wirft Pydantic einen klaren Fehler.

class ModelSettings(BaseModel):
    embedding_id: str
    reranker_id: str
    ollama_llm: str
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
    use_reranker: bool
    retrieval_top_k: int
    default_retrieval_top_k: int
    min_chunks_to_llm: int
    max_chunks_to_llm: int
    min_absolute_score_threshold: float
    min_chunks_for_gap_detection: int
    gap_detection_factor: float
    small_epsilon: float

class AppSettings(BaseModel):
    """Das Haupt-Einstellungsmodell, das alle anderen Einstellungs-Klassen aggregiert."""
    models: ModelSettings
    database: DatabaseSettings
    processing: ProcessingSettings
    pipeline: PipelineSettings

def load_settings(config_path: str = "config.yaml") -> AppSettings:
    """
    Lädt die Konfiguration aus einer YAML-Datei und validiert sie mit Pydantic.
    """
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

# --- Globales Einstellungsobjekt ---
# Die Einstellungen werden einmal beim Start geladen. Andere Module können dieses einzelne 'settings'-Objekt importieren.
# Dieser Ansatz stellt sicher, dass die Konfiguration nur einmal geladen und validiert wird.
try:
    settings = load_settings()
    print("Konfiguration erfolgreich geladen und validiert.")
except Exception:
    # Die Anwendung kann ohne gültige Einstellungen nicht laufen, daher bei Fehler beenden.
    print("Beende aufgrund eines Konfigurationsfehlers.")
    exit(1)