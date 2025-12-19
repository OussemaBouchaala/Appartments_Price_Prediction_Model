from utils import load_config

# Load configuration
_config = load_config()
#print("Configuration loaded:", _config)

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
