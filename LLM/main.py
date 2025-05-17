import pickle
import logging
import torch
import torch.nn as nn
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Loading TF-IDF tokenizer and model")

# Load the TF-IDF tokenizer
with open('LLM/tfidf_tokenizer.pkl', 'rb') as f:
    tfidf_tokenizer = pickle.load(f)

# Define the same model architecture used in training
class FeedforwardNN(nn.Module):
    def __init__(self, input_dim):
        super(FeedforwardNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Initialize and load model
input_dim = 5000  # Must match tfidf_vectorizer max_features
model = FeedforwardNN(input_dim)
model.load_state_dict(torch.load('LLM/text_classification_model.pth'))
model.eval()

# Function to preprocess text and make predictions
def predict_outcome(text):
    logger.debug("Predicting outcome using LLM model")

    # Transform input text to TF-IDF features
    text_features = tfidf_tokenizer.transform([text]).toarray()
    text_tensor = torch.tensor(text_features, dtype=torch.float32)

    # Run inference
    with torch.no_grad():
        prediction = model(text_tensor).item()
        predicted_label = int(prediction > 0.5)

    return predicted_label
