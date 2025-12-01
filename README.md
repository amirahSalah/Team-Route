
# Repository

This repository contains the main Streamlit app in `src/streamlit_app.py`.

## How to Run This Streamlit Project

### 1. Navigate to the project root
```sh
# All platforms (macOS/Linux/Windows)
cd "/path-to-the-project-root/Team-Route"
```

### 2. Create a virtual environment
```sh
# macOS/Linux:
python3 -m venv .venv
# Windows:
python -m venv .venv
```

### 3. Activate the virtual environment
```sh
# macOS/Linux:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate
```

### 4. Verify Python installation
```sh
# macOS/Linux:
which python
# Windows:
where python

# Both platforms:
python -V
```

### 5. Upgrade pip and install dependencies
```sh
# Both platforms (macOS/Linux/Windows):
python -m pip install --upgrade pip
python -m pip install streamlit pandas numpy joblib plotly
```

### 6. Run the Streamlit app
```sh
# Both platforms (macOS/Linux/Windows):
streamlit run src/streamlit_app.py
```
