import hashlib
import string
from sentence_transformers import SentenceTransformer, util
import torch
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
import nltk

nltk.download('punkt_tab')
nltk.download('stopwords')

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    translator = str.maketrans('', '', string.punctuation)

    sentences = sent_tokenize(text)
    preprocessed_sentences = []

    for sentence in sentences:
        sentence = sentence.lower().translate(translator)
        words = [word for word in sentence.split() if word not in stop_words]
        preprocessed_sentences.append(" ".join(words))

    return preprocessed_sentences

def calculate_text_hash(text):
    return hashlib.sha256(text.strip().encode('utf-8')).hexdigest()

def calculate_similarity(text1, text2):
    hash1 = calculate_text_hash(text1)
    hash2 = calculate_text_hash(text2)

    if hash1 == hash2:
        return 1.0, None, sent_tokenize(text1), sent_tokenize(text2)  # 100% similarity, no need for embeddings

    model = SentenceTransformer('paraphrase-mpnet-base-v2', device='cuda' if torch.cuda.is_available() else 'cpu')

    sentences1 = preprocess_text(text1)
    sentences2 = preprocess_text(text2)

    embeddings1 = model.encode(sentences1, convert_to_tensor=True)
    embeddings2 = model.encode(sentences2, convert_to_tensor=True)

    cosine_similarities = util.pytorch_cos_sim(embeddings1, embeddings2)
    max_similarities = torch.max(cosine_similarities, dim=1).values
    avg_similarity = torch.mean(max_similarities).item()

    return avg_similarity, cosine_similarities, sentences1, sentences2

def highlight_plagiarism(cosine_similarities, sentences_to_highlight, sentences_to_compare):
    if cosine_similarities is None:
        return "[RED] Document is an exact match [END]\n"

    highlighted_text = ""
    for i, sentence in enumerate(sentences_to_highlight):
        if i < cosine_similarities.size(0):
            max_similarity = torch.max(cosine_similarities[i]).item()
            if max_similarity >= 0.9:
                highlighted_text += f"[RED] {sentence.strip()} [END]\n"
            elif 0.75 <= max_similarity < 0.9:
                highlighted_text += f"[YELLOW] {sentence.strip()} [END]\n"
            else:
                highlighted_text += f"{sentence.strip()}\n"
    return highlighted_text
