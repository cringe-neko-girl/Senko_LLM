import os
import random
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = {"corpus": [], "vectors": None, "vectorizer": None}
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_model.eval()
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

def update_vectors():
    if model["corpus"]:
        vectorizer = TfidfVectorizer().fit(model["corpus"])
        model["vectors"] = vectorizer.transform(model["corpus"])
        model["vectorizer"] = vectorizer

def load_model():
    path = "models/model.bin"
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                loaded = pickle.load(f)
                if "corpus" in loaded:
                    model["corpus"] = loaded["corpus"]
                    if "vectors" in loaded:
                        model["vectors"] = loaded["vectors"]
                    if "vectorizer" in loaded:
                        model["vectorizer"] = loaded["vectorizer"]
        except Exception as e:
            print(f"Error loading model: {e}")
    return model

def save_model(_model):
    os.makedirs("models", exist_ok=True)
    try:
        with open("models/model.bin", "wb") as f:
            pickle.dump({
                "corpus": _model["corpus"],
                "vectors": _model["vectors"],
                "vectorizer": _model["vectorizer"]
            }, f)
    except Exception as e:
        print(f"Error saving model: {e}")

def limit_word_count(text, max_words=5000):
    words = text.split()
    return ' '.join(words[:max_words]) if len(words) > max_words else text

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

def pick_diverse_segment(text):
    segments = sentence_start_segments(text)
    unique_segments = []
    for seg in segments:
        if all(jaccard_similarity(seg, other) < 0.5 for other in unique_segments):
            unique_segments.append(seg)
    return random.choice(unique_segments or segments)

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
            num_beams=5,
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

def generate_reply(query):
    if not model["corpus"] or len(model["corpus"]) < 2:
        return "I don't know enough yet. Please upload more knowledge."
    if not model["vectorizer"] or model["vectors"] is None:
        update_vectors()

    query_vec = model["vectorizer"].transform([query])
    sims = cosine_similarity(query_vec, model["vectors"]).flatten()

    top_indices = sims.argsort()[::-1]
    top_indices = [i for i in top_indices if sims[i] > 0.2][:5]  # Limit to top 5 responses for diversity

    if not top_indices:
        return "Sorry, I don't know enough about that."

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
    best_reply = scored[0][1] if scored else "Hmm... I'm not sure how to answer that."

    return limit_word_count(best_reply)
