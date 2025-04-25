import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

model = {"corpus": [], "vectors": None, "vectorizer": None}

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
                    update_vectors()
        except Exception as e:
            print(f"Error loading model: {e}")
    return model

def save_model(_model):
    os.makedirs("models", exist_ok=True)
    try:
        with open("models/model.bin", "wb") as f:
            pickle.dump({"corpus": _model["corpus"]}, f)
    except Exception as e:
        print(f"Error saving model: {e}")

def generate_reply(query):
    if not model["corpus"] or len(model["corpus"]) < 2:
        return "I don't know enough yet. Please upload more knowledge."

    vectorizer = model.get("vectorizer")
    if not vectorizer or model["vectors"] is None:
        update_vectors()
        vectorizer = model.get("vectorizer")

    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, model["vectors"]).flatten()

    idx = sims.argmax()
    if sims[idx] < 0.2:
        return "Sorry, I don't know enough about that."

    return model["corpus"][idx]
