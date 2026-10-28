from pathlib import Path

from utils import load_config

# Load configuration
_config = load_config()
#print("Configuration loaded:", _config)

# config.json paths are relative to app/; resolve them so the app works from any working directory
_APP_DIR = Path(__file__).parent
_config['paths'] = {k: str((_APP_DIR / v).resolve()) for k, v in _config['paths'].items()}
_config['data_files'] = {k: str((_APP_DIR / v).resolve()) for k, v in _config['data_files'].items()}

# Path variables
MODELS_DIR = _config['paths']['models_dir']
DATA_DIR = _config['paths']['data_dir']
NOTEBOOKS_DIR = _config['paths']['notebooks_dir']

# Model paths
MODEL_RF_OPTIMIZED = _config['paths']['model_rf_optimized']
SCALER = _config['paths']['scaler']
MODEL_COLUMNS = _config['paths']['model_columns']
MODEL_RF2 = _config['paths']['model_rf2']
FEATURES = _config['paths']['features']

# Data file paths
DATA_FILES = _config['data_files']
