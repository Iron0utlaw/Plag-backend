import pickle
from tensorflow.keras.models import load_model
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Loading TF-IDF tokenizer and model")

# Load the TF-IDF tokenizer``
with open('LLM/tfidf_tokenizer.pkl', 'rb') as f:
    tfidf_tokenizer = pickle.load(f)

# Load the trained model
loaded_model = load_model('LLM/text_classification_model.h5')

# Function to preprocess text and make predictions
def predict_outcome(text):
    logger.debug("Predicting outcome using LLM model")

    # Preprocess the text using the loaded tokenizer
    text_features = tfidf_tokenizer.transform([text])
    
    # Make predictions using the loaded model
    predictions = loaded_model.predict(text_features)
    
    # Return the predicted outcome (1 for positive, 0 for negative)
    return int(predictions[0][0])
