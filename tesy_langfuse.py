from langfuse import Langfuse

try:
    client = Langfuse(public_key="test", secret_key="test")
    print("Import Langfuse działa!")
except ImportError as e:
    print(f"Błąd importu: {e}")