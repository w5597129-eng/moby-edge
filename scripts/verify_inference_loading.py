
import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "src"))
from src.inference_worker import build_default_engine, current_timestamp_ns, WindowMessage

def verify():
    print("Building Inference Engine...")
    try:
        engine = build_default_engine()
        print("Engine built successfully.")
        
        # Check if MLP model is loaded
        mlp_runner = next((r for r in engine.runners if r.config.name == "mlp_classifier"), None)
        if mlp_runner and mlp_runner.model is not None:
             print(f"Verified: MLP Classifier loaded. Type: {type(mlp_runner.model)}")
        else:
             print("FAILED: MLP Classifier not loaded.")
             sys.exit(1)

        print("\nSUCCESS: Inference Worker is ready with the new model.")
        
    except Exception as e:
        print(f"FAILED to build engine: {e}")
        sys.exit(1)

if __name__ == "__main__":
    verify()
