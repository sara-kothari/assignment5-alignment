import numpy as np
from cs336_alignment.modal_utils import DATA_PATH, app
from pathlib import Path
import glob

@app.function()
def upload_file(data: bytes, filename: str):
    output_path = DATA_PATH / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(data)
    print(f"Saved to {output_path}")

@app.local_entrypoint()
def modal_main():
    for filename in glob.glob("cs336_alignment/prompts/*.prompt"):
        data = Path(filename).read_bytes()
        upload_file.remote(data, filename)