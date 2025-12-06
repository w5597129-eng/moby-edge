
import sys
import os

# Ensure src is in path to find the real module
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Import from the specific file in src to avoid module name collision
# We use importlib to be specific
import importlib.util
spec = importlib.util.spec_from_file_location("src_feature_extractor", os.path.join(src_dir, "feature_extractor.py"))
module = importlib.util.module_from_spec(spec)
sys.modules["src_feature_extractor"] = module
spec.loader.exec_module(module)

# Expose symbols
extract_features = module.extract_features
FEATURE_CONFIG_V19 = module.FEATURE_CONFIG_V19
FEATURE_CONFIG_V18 = FEATURE_CONFIG_V19
