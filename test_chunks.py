from app import build_chunks_for_file
from pathlib import Path
import time, traceback

p = r"D:\rag_project\uploads\cf181d394bcc4c599fe4b4368df65a1e.pdf"
print("TEST: build_chunks_for_file on:", p)

try:
    t0 = time.time()
    chunks = build_chunks_for_file(p, max_images_ocr=1)
    dt = time.time() - t0
    print(f"-> elapsed: {dt:.2f}s    chunks count: {len(chunks)}\n")

    for i, c in enumerate(chunks[:3]):
        print(f"--- chunk {i} (len: {len(c)}) ---")
        print(c[:700].replace("\n"," "))
        print("\n")

except Exception:
    print("ERROR:")
    traceback.print_exc()
