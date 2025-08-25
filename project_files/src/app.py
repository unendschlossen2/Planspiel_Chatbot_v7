import streamlit as st
import time
import re
from typing import Set

# --- Import all your project's modules ---
from helper.settings import settings
from helper.load_gpu import load_gpu
from embeddings.embedding_generator import load_embedding_model
from retrieval.reranker import load_reranker_model, gap_based_rerank_and_filter
from vector_store.vector_store_manager import create_and_populate_vector_store
from retrieval.retriever import query_vector_store, embed_query
from generation.conversation_handler import condense_conversation
from generation.query_expander import expand_user_query
from generation.llm_answer_generator import generate_llm_answer


# --- Caching Functions for Model and DB Loading ---
# @st.cache_resource tells Streamlit to run this function only once.
# This is CRUCIAL for performance.

@st.cache_resource
def load_processing_device():
    """Loads the processing device (GPU or CPU) once."""
    try:
        return str(load_gpu())
    except RuntimeError as e:
        st.error(f"GPU Ladefehler: {e}. Wechsle zu CPU.")
        return "cpu"

@st.cache_resource
def load_models_and_db(device):
    """Loads all models and the database connection once."""
    embedding_model = load_embedding_model(settings.models.embedding_id, device)
    reranker_model = None
    if settings.pipeline.use_reranker:
        reranker_model = load_reranker_model(settings.models.reranker_id, device=device)

    db_collection = create_and_populate_vector_store(
        chunks_with_embeddings=[],
        db_path=settings.database.persist_path,
        collection_name=settings.database.collection_name,
        force_rebuild_collection=False
    )
    return embedding_model, reranker_model, db_collection

# --- Main Chatbot Logic ---

def get_bot_response(user_query: str, chat_history: list):
    """
    This function contains the full RAG pipeline logic.
    It takes a user query and chat history and returns the bot's response.
    """
    # Load device, models, and DB from cache
    device = load_processing_device()
    embedding_model, reranker_model, db_collection = load_models_and_db(device)

    if not db_collection:
        return "Fehler: Datenbank-Kollektion konnte nicht geladen werden.", {}

    # --- Conversational Memory ---
    last_query = chat_history[-2]['content'] if len(chat_history) > 1 else None
    last_response = chat_history[-1]['content'] if len(chat_history) > 1 and chat_history[-1]['role'] == 'assistant' else ""

    condensed_query = user_query
    if settings.pipeline.enable_conversation_memory and last_query:
        condensed_query = condense_conversation(
            model_name=settings.models.condenser_model_id,
            last_query=last_query,
            last_response=last_response,
            new_query=user_query
        )

    # --- Query Expansion ---
    expanded_query = condensed_query
    if settings.pipeline.enable_query_expansion:
        expanded_query = expand_user_query(
            user_query=condensed_query,
            model_name=settings.models.query_expander_id,
            char_threshold=settings.pipeline.query_expansion_char_threshold
        )

    # --- Retrieval ---
    query_embedding = embed_query(embedding_model, expanded_query)
    retrieved_docs = query_vector_store(db_collection, query_embedding, settings.pipeline.retrieval_top_k)

    # --- Reranking ---
    final_docs = retrieved_docs
    if settings.pipeline.use_reranker and reranker_model:
        final_docs = gap_based_rerank_and_filter(
            user_query=expanded_query,
            initial_retrieved_docs=retrieved_docs,
            reranker_model=reranker_model,
            min_absolute_rerank_score_threshold=settings.pipeline.min_absolute_score_threshold,
            min_chunks_to_llm=settings.pipeline.min_chunks_to_llm,
            max_chunks_to_llm=settings.pipeline.max_chunks_to_llm,
            min_chunks_for_gap_detection=settings.pipeline.min_chunks_for_gap_detection,
            gap_detection_factor=settings.pipeline.gap_detection_factor
        )

    if not final_docs:
        return "Ich konnte leider keine relevanten Informationen zu Ihrer Anfrage im Handbuch finden.", {}

    # --- Answer Generation & Formatting ---
    llm_answer_generator, citation_map = generate_llm_answer(
        user_query=condensed_query,
        retrieved_chunks=final_docs,
        ollama_model_name=settings.models.ollama_llm,
        ollama_options=settings.models.ollama_options,
    )

    raw_response = "".join([chunk for chunk in llm_answer_generator])

    # --- Citation Handling ---
    used_source_ids: Set[int] = set()
    citation_regex = re.compile(r'\[Source ID: (\d+)]')

    matches = citation_regex.finditer(raw_response)
    for match in matches:
        used_source_ids.add(int(match.group(1)))

    formatted_response = citation_regex.sub(r'[\1]', raw_response).strip()

    # --- Source Formatting ---
    sources_text = ""
    if used_source_ids and citation_map:
        sources_list = []
        for source_id in sorted(list(used_source_ids)):
            if source_id in citation_map:
                info = citation_map[source_id]
                sources_list.append(f"- [{source_id}] **{info['filename']}**, Abschnitt: *{info['header']}*")
        sources_text = "\n".join(sources_list)

    return formatted_response, sources_text


# --- Streamlit UI ---

st.set_page_config(page_title="TOPSIM RAG Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 TOPSIM RAG Chatbot")

# Sidebar for configuration and controls
with st.sidebar:
    st.header("Konfiguration")
    if st.button("Neuer Chat"):
        st.session_state.messages = []

    st.info(f"**LLM:** `{settings.models.ollama_llm}`")
    st.info(f"**Reranker:** `{'Aktiviert' if settings.pipeline.use_reranker else 'Deaktiviert'}`")
    st.info(f"**Speichermodus:** `{'Low-VRAM' if settings.system.low_vram_mode else 'Standard'}`")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input field
if prompt := st.chat_input("Stellen Sie Ihre Frage an das TOPSIM Handbuch..."):
    # Add and display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display bot response
    with st.chat_message("assistant"):
        with st.spinner("Bot denkt nach..."):
            response_text, sources = get_bot_response(prompt, st.session_state.messages)

            # Use the "fake stream" for better user experience
            placeholder = st.empty()
            full_response = ""
            for char in response_text:
                full_response += char
                placeholder.markdown(full_response + "▌")
                time.sleep(0.01)
            placeholder.markdown(full_response)

            if sources:
                with st.expander("**Quellen**"):
                    st.markdown(sources)

    # Add bot response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_text + ("\n\n**Quellen:**\n" + sources if sources else "")})