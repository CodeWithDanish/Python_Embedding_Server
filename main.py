from flask import Flask, request, jsonify
import requests
import json  

app = Flask(__name__)

# /embed endpoint remains the same — for embeddings
from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

@app.route('/embed', methods=['POST'])
def embed_text():
    data = request.get_json()
    text = data.get('text')
    if not text:
        return jsonify({'error': 'Missing "text" in request'}), 400

    try:
        embedding = embedding_model.encode(text).tolist()
        return jsonify({'embedding': embedding})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 🆕 New endpoint to generate answers using TinyLlama via Ollama
@app.route('/generate', methods=['POST'])
def generate_answer():
    data = request.get_json()
    prompt = data.get('prompt')
    if not prompt:
        return jsonify({'error': 'Missing "prompt" in request'}), 400

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "tinyllama", "prompt": prompt},
            stream=True
        )

        if response.status_code != 200:
            return jsonify({"error": f"Failed with status {response.status_code}"}), 500

        generated_text = ""

        for line in response.iter_lines():
            if line:
                try:
                    json_data = json.loads(line.decode("utf-8"))
                    generated_text += json_data.get("response", "")
                except Exception as e:
                    continue  # Ignore lines that can't be parsed

        return jsonify({"response": generated_text})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
# Run the app
if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
