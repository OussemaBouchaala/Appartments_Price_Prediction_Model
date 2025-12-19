import json
from pathlib import Path


def load_config():
    """Load configuration from config.json file."""
    config_path = Path(__file__).parent.parent / "config.json"
    #print(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        return json.load(f)
