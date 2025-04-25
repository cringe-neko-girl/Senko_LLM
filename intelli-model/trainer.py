from model import model, update_vectors

def train_model(text: str):
    text = text.strip()
    if text:
        model["corpus"].append(text)
        update_vectors()
