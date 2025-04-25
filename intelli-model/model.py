import os
import pickle
import json, random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Path configuration
MODEL_PATH = "models/model.bin"
DATA_PATH = "data/knowledge_store.json"

model = {"corpus": [], "vectors": None, "vectorizer": None}
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_model.eval()
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

def update_vectors():
    if model["corpus"]:
        if model["vectorizer"] is None:  # Only re-fit if no vectorizer exists
            vectorizer = TfidfVectorizer().fit(model["corpus"])
            model["vectors"] = vectorizer.transform(model["corpus"])
            model["vectorizer"] = vectorizer

def load_model():
    # Try loading the knowledge store and model together
    knowledge_data = load_knowledge()

    # Load model data from model.bin if it exists
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                loaded = pickle.load(f)
                if "corpus" in loaded:
                    model["corpus"] = loaded["corpus"]
                if "vectors" in loaded:
                    model["vectors"] = loaded["vectors"]
                if "vectorizer" in loaded:
                    model["vectorizer"] = loaded["vectorizer"]
        except Exception as e:
            print(f"Error loading model: {e}")
    
    # Load data from knowledge.json into model
    if knowledge_data:
        for entry in knowledge_data:
            content = entry["content"]
            if content not in model["corpus"]:
                model["corpus"].append(content)
        update_vectors()
    
    return model

def save_model(_model):
    # Save the model as a .bin file
    os.makedirs("models", exist_ok=True)
    try:
        with open(MODEL_PATH, "wb") as f:
            pickle.dump({
                "corpus": _model["corpus"],
                "vectors": _model["vectors"],
                "vectorizer": _model["vectorizer"]
            }, f)
    except Exception as e:
        print(f"Error saving model: {e}")

def save_knowledge(knowledge_data):
    # Save the knowledge store as a JSON file
    os.makedirs("data", exist_ok=True)
    try:
        with open(DATA_PATH, 'w', encoding='utf8') as f:
            json.dump(knowledge_data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Error saving knowledge data: {e}")

def load_knowledge():
    # Load knowledge store from knowledge.json
    if os.path.exists(DATA_PATH):
        try:
            with open(DATA_PATH, 'r', encoding='utf8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("Warning: knowledge store JSON is malformed. Starting fresh.")
    return []

def generate_reply(query):
    if not model["corpus"] or len(model["corpus"]) < 2:
        return "I don't have enough knowledge yet. Please upload more knowledge."

    if not model["vectorizer"] or model["vectors"] is None:
        update_vectors()

    query_vec = model["vectorizer"].transform([query])
    sims = cosine_similarity(query_vec, model["vectors"]).flatten()

    top_indices = sims.argsort()[::-1]
    top_indices = [i for i in top_indices if sims[i] > 0.2][:5]  # Limit to top 5 responses for diversity

    if not top_indices:
        return "Sorry, I don't have enough data on that topic."

    selected_sources = [model["corpus"][i] for i in top_indices]
    candidates = []
    for source in selected_sources:
        candidates.extend(natural_response(source, query))

    scored = []
    for cand in candidates:
        if not cand.strip():
            continue
        cand_vec = model["vectorizer"].transform([cand])
        score = cosine_similarity(query_vec, cand_vec).flatten()[0]
        scored.append((score, cand))

    scored.sort(reverse=True, key=lambda x: x[0])
    best_reply = scored[0][1] if scored else "Hmm, I don't know how to answer that."

    return best_reply

def natural_response(response, query):
    trimmed = pick_diverse_segment(response[:1500])
    encoded = gpt2_tokenizer(trimmed, return_tensors="pt", padding=True)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    try:
        output = gpt2_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=min(input_ids.shape[1] + 150, 1024),
            num_return_sequences=3,  # Reduce to the best 3 responses
            no_repeat_ngram_size=2,
            pad_token_id=gpt2_tokenizer.eos_token_id,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.2,
            length_penalty=1.0,
            num_beams=3,  # Reduce number of beams for speed
            do_sample=True,
            early_stopping=True
        )
        responses = [gpt2_tokenizer.decode(out, skip_special_tokens=True) for out in output]
        unique = []
        for r in responses:
            lines = set()
            is_repetitive = False
            for line in r.split('.'):
                line = line.strip()
                if line in lines:
                    is_repetitive = True
                    break
                lines.add(line)
            if not is_repetitive and r not in unique:
                unique.append(r)
        return unique if unique else responses
    except Exception as e:
        print(f"Error generating response: {e}")
        return ["Sorry, there was an error generating the response."]

def pick_diverse_segment(text):
    segments = sentence_start_segments(text)
    unique_segments = []
    for seg in segments:
        if all(jaccard_similarity(seg, other) < 0.5 for other in unique_segments):
            unique_segments.append(seg)
    return random.choice(unique_segments or segments)

def sentence_start_segments(text, segment_length=120):
    sentences = text.split('. ')
    segments = []
    for i in range(len(sentences)):
        seg = '. '.join(sentences[i:i+3])
        if len(seg.split()) >= segment_length // 2:
            segments.append(seg.strip() + ('' if seg.endswith('.') else '.'))
    return segments or [text]

def jaccard_similarity(a, b):
    sa, sb = set(a.lower().split()), set(b.lower().split())
    return len(sa & sb) / max(1, len(sa | sb))

def update_vectors():
    if model["corpus"]:
        if model["vectorizer"] is None:  # Only re-fit if no vectorizer exists
            vectorizer = TfidfVectorizer().fit(model["corpus"])
            model["vectors"] = vectorizer.transform(model["corpus"])
            model["vectorizer"] = vectorizer
