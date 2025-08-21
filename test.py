import time
import gc
import torch
from sentence_transformers import SentenceTransformer

# --- CONFIGURATION ---
# Use the same model ID and device detection as your main project.
EMBEDDING_MODEL_ID = "infly/inf-retriever-v1-1.5b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def check_vram():
    """Helper function to print current VRAM usage."""
    if DEVICE == "cuda":
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"VRAM Usage: {allocated:.2f} GB allocated / {reserved:.2f} GB reserved")
    else:
        print("Running on CPU, no VRAM to check.")

def main():
    """
    Main function to load and unload the model for testing.
    """
    print("--- VRAM Unloading Test Script ---")
    print(f"Using device: {DEVICE}")
    if DEVICE != "cuda":
        print("Exiting because no CUDA GPU was detected.")
        return

    print("\n--- Phase 1: Loading Model ---")
    check_vram()
    print(f"Loading model '{EMBEDDING_MODEL_ID}' onto the GPU...")

    # Load the model
    model = SentenceTransformer(EMBEDDING_MODEL_ID, device=DEVICE)

    # Run a dummy inference to ensure it's fully loaded
    model.encode("test")

    print("\n✅ Model is now loaded into VRAM.")
    check_vram()
    input("--> Press Enter to attempt unloading the model...")

    print("\n--- Phase 2: Unloading Model ---")

    # Method 1: Delete the Python object
    print("Step 1: Deleting model variable (`del model`)...")
    model = None

    # Method 2: Call Python's garbage collector
    print("Step 2: Calling garbage collector (`gc.collect()`)...")
    gc.collect()

    # Method 3: Tell PyTorch to release cached memory
    print("Step 3: Emptying PyTorch CUDA cache (`torch.cuda.empty_cache()`)...")
    torch.cuda.empty_cache()

    print("\n✅ Unloading sequence complete.")
    check_vram()
    print("--> Check your VRAM now to see the result. The script will exit.")
    time.sleep(10)


if __name__ == "__main__":
    main()