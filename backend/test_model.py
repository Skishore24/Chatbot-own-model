import sys
import torch

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

MODEL_PATH = "genkit-model/model_v6.pt"

try:
    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False
    )

    print("\n[SUCCESS] MODEL FILE LOADED SUCCESSFULLY")
    print("Type:", type(checkpoint))

    if isinstance(checkpoint, dict):
        print("\nCheckpoint keys:")
        for key in checkpoint.keys():
            print("-", key)

except Exception as e:
    print("\n[ERROR] MODEL FILE FAILED TO LOAD")
    print(type(e).__name__)
    print(e)
    