from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from LLM.main import predict_outcome
from Doc_sim.main import calculate_similarity, highlight_plagiarism
from utils import extract_text
import itertools
from flask_cors import CORS
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app) 

@app.route('/llm_detect', methods=['POST'])
def llm_detect():
    logger.info("Received request for LLM detection")
    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': 'No files uploaded'}), 400

    results = {}
    for file in files:
        try:
            text = extract_text(file)
            label = predict_outcome(text)
            results[secure_filename(file.filename)] = label
        except Exception as e:
            results[secure_filename(file.filename)] = f"Error: {str(e)}"

    return jsonify(results)

@app.route('/doc_similarity', methods=['POST'])
def doc_similarity():
    logger.info("Received request for LLM detection")
    files = request.files.getlist('files')
    if not files or len(files) < 2:
        return jsonify({'error': 'At least two files are required'}), 400

    texts = {}
    for file in files:
        try:
            texts[secure_filename(file.filename)] = extract_text(file)
        except Exception as e:
            return jsonify({'error': f"{file.filename}: {str(e)}"}), 500

    comparisons = {}
    file_pairs = itertools.combinations(texts.items(), 2)

    for (name1, text1), (name2, text2) in file_pairs:
        try:
            score, cos_sim, s1, s2 = calculate_similarity(text1, text2)
            highlighted = highlight_plagiarism(cos_sim, s2, s1)
            comparisons[f"{name1} vs {name2}"] = {
                'similarity_score': f"{score * 100:.2f}%",
                'highlighted_text': highlighted
            }
        except Exception as e:
            comparisons[f"{name1} vs {name2}"] = {'error': str(e)}

    return jsonify(comparisons)

if __name__ == '__main__':
    app.run(debug=True)
