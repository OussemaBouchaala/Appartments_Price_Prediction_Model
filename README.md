# Tunisia Real Estate Price Predictor 🏠

A machine learning-powered web application for predicting real estate prices in Tunisia using property features like surface area, bedrooms, bathrooms, and location.

## Features

- **Interactive UI**: Built with Streamlit for easy property price predictions
- **ML-Powered**: Uses Random Forest model trained on Tunisia real estate data
- **Dynamic Cities**: Automatically loads available cities from the model
- **Real-time Predictions**: Instant price estimates based on input features

## Project Structure

```
Mini_project/
├── app/
│   ├── config.py          # Configuration variables
│   ├── utils.py           # Config loader utility
│   ├── streamlit.py       # Main Streamlit app
│   └── streamlit_v0.py    # Alternative version
├── data/                  # Dataset files
├── models/                # Trained models and scalers
├── notebooks/             # Jupyter notebooks for training
├── config.json            # Centralized path configuration
└── requirements.txt       # Python dependencies
```

## Installation

1. Clone the repository
2. Create virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the Streamlit app:
```bash
cd app
streamlit run streamlit.py
```

Enter property details (surface area, bedrooms, bathrooms, city) and get instant price predictions in TND.

## Model

The model predicts log-transformed prices using:
- **Features**: Surface area, number of bedrooms, bathrooms, city
- **Algorithm**: Optimized Random Forest Regressor
- **Preprocessing**: StandardScaler for numerical features, One-Hot encoding for cities

## License

MIT License
