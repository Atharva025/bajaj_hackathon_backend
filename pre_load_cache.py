# pre_load_cache.py

from main import get_or_create_vectorstore # Reuse the caching logic from the main app
import time

# IMPORTANT: Add all known hackathon document URLs to this list
KNOWN_DOCUMENT_URLS = [
    "https://hackrx.in/policies/BAJHLIP23020V012223.pdf",
    "https://hackrx.in/policies/CHOTGDP23004V012223.pdf",
    "https://hackrx.in/policies/EDLHLGA23009V012223.pdf",
    "https://hackrx.in/policies/HDFHLIP23024V072223.pdf",
    "https://hackrx.in/policies/ICIHLIP22012V012223.pdf",
    "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "https://hackrx.blob.core.windows.net/assets/hackrx_6/policies/HDFHLIP23024V072223.pdf?sv=2023-01-03&st=2025-07-30T06%3A46%3A49Z&se=2025-09-01T06%3A46%3A00Z&sr=c&sp=rl&sig=9szykRKdGYj0BVm1skP%2BX8N9%2FRENEn2k7MQPUp33jyQ%3D"
]

if __name__ == "__main__":
    print("Starting cache pre-loading process for all known documents...")
    start_time = time.time()
    
    for url in KNOWN_DOCUMENT_URLS:
        try:
            print(f"\n--- Processing: {url} ---")
            get_or_create_vectorstore(url)
        except Exception as e:
            print(f"Failed to process {url}. Error: {e}")
            
    end_time = time.time()
    print(f"\n✅ Cache pre-loading finished in {end_time - start_time:.2f} seconds.")
