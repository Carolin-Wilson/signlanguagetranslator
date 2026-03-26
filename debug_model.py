import traceback, json, h5py

# First print what the patched config looks like
with h5py.File("model/sign_language_model.h5", "r") as f:
    raw = f.attrs.get("model_config")
    raw_str = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
    cfg = json.loads(raw_str)
    # Print just the first layer config to inspect
    try:
        layers = cfg["config"]["layers"]
        print("First layer config:", json.dumps(layers[0], indent=2))
    except Exception as e:
        print("Config structure:", json.dumps(cfg, indent=2)[:2000])

print("\n--- Attempting load ---")
try:
    from keras.models import load_model
    model = load_model("model/sign_language_model.h5")
    print("SUCCESS:", model.input_shape)
except Exception:
    traceback.print_exc()
