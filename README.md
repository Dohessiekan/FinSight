# FinSight

## Description
FinSight is a machine learning and natural language processing (NLP) project for sentiment analysis and classification of SMS messages, with a focus on spam detection. The project leverages modern NLP techniques, including TF-IDF vectorization, Word2Vec embeddings, and deep learning models (LSTM/GRU), to analyze and visualize text data. It includes exploratory data analysis, feature engineering, and model evaluation with various machine learning algorithms.

## GitHub Repository
This project currently does not have a public GitHub repository. To create one, visit [GitHub](https://github.com/) and follow the instructions to initialize a new repository. If you already have a repository, add the remote link here.

## Environment Setup

### 1. Clone the repository (if available)
```powershell
git clone <your-repo-url>
cd FinSight
```

### 2. Install Python (Recommended: Python 3.8+)
Download and install Python from [python.org](https://www.python.org/downloads/).

### 3. Create and activate a virtual environment (optional but recommended)
```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 4. Install dependencies
Open a terminal in the project directory and run:
```powershell
pip install -r requirements.txt
```
If `requirements.txt` is not present, install the main libraries manually:
```powershell
pip install numpy pandas matplotlib seaborn scikit-learn nltk emoji gensim xgboost lightgbm catboost tensorflow keras wordcloud tsfresh
```

### 5. Download NLTK resources (run in Python shell or notebook)
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
```

### 6. (Colab only) Mount Google Drive
If running in Google Colab, mount your drive to access datasets:
```python
from google.colab import drive
drive.mount('/content/drive')
```

## Project Structure
- `Models/capstone (1).ipynb`: Main Jupyter notebook with all code, analysis, and model training.
- `Models/sentiment_model.pkl`: Example of a saved model.
- `README.md`: Project documentation.

## Designs
No Figma mockups, circuit diagrams, or screenshots were found in the repository. To add designs:
- Export Figma mockups/screenshots as PNG/JPG and place them in the project folder.
- For circuit diagrams, use tools like draw.io and export as image or PDF.
- To display images in the README, use:
  ```markdown
  ![Screenshot](path/to/image.png)
  ```

## Deployment Plan
1. **Model Export**: Trained models are saved as `.pkl` (for scikit-learn) or `.keras`/`.h5` (for TensorFlow/Keras).
2. **API/Service (Optional)**: Wrap the model in a REST API using Flask or FastAPI for production use.
3. **Web App (Optional)**: Build a simple web interface using Streamlit, Flask, or Django for user interaction.
4. **Colab/Notebook**: For research and demo, run the notebook in Jupyter or Google Colab.
5. **Documentation**: Keep this README updated with setup, usage, and results.

## Required Model Files
The API requires several trained model files that are not included in the repository for size and security reasons. You will need to place these files in the correct locations:

### Core Model Files (Root Directory)
- `spam_classifier.h5` - The trained TensorFlow model for spam detection
- `tokenizer.pkl` - The fitted tokenizer for text preprocessing
- `label_encoder.pkl` - The label encoder for class mapping
- `max_len.pkl` - Configuration for sequence padding

### Model Artifacts Location
```plaintext
FinSight/
├── API/
│   └── main.py
├── spam_classifier.h5
├── tokenizer.pkl
├── label_encoder.pkl
└── max_len.pkl
```

### Getting the Model Files
1. Option 1 - Train the Model:
   - Run the Jupyter notebook `Models/capstone (1).ipynb`
   - The notebook will generate all required model files
   - Move the files to the root directory

2. Option 2 - Download Pre-trained Models:
   - Request access to the pre-trained models from the project maintainers
   - Place the files in the root directory as shown above

### Validating Model Files
After placing the model files, verify their presence:
```powershell
Get-ChildItem -Path . -Filter *.{h5,pkl}
```

You should see:
- spam_classifier.h5
- tokenizer.pkl
- label_encoder.pkl
- max_len.pkl

If any files are missing, the API will fail to start with a FileNotFoundError.

### Note on Model Security
- The model files contain learned patterns from the training data
- Do not commit these files to version control
- Keep backups of the model files in a secure location
- Consider implementing model versioning for production deployments

---
*Last updated: June 9, 2025*