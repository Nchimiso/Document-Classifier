import os
import glob
import time
import fitz
import numpy as np
import gradio as gr
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# Load dataset from /data
def load_custom_dataset(data_dir="./data"):
    texts = []
    labels = []
    label_names = sorted(os.listdir(data_dir))
    for label in label_names:
        class_dir = os.path.join(data_dir, label)
        if os.path.isdir(class_dir):
            for file_path in glob.glob(os.path.join(class_dir, "*.txt")):
                if file_path.endswith(".txt"):
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                elif file_path.endswith(".pdf"):
                    doc = fitz.open(file_path)
                    text = "\n".join([page.get_text() for page in doc])
                    doc.close()
                if text.strip():
                    texts.append(text)
                    labels.append(label)
    return texts, labels, label_names

# training function
def train():
    global clf
    global vectorizer
    global label_names
    global acc

    texts, labels, label_names = load_custom_dataset()

    print(f"Loaded {len(texts)} documents across {len(set(labels))} categories")

    # Split into training and testing
    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=42
    )

    # Vectorizer creation 
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, min_df=1, stop_words="english")
    X_train = vectorizer.fit_transform(X_train_texts)

    # Extract test data features using vectorizer
    X_test = vectorizer.transform(X_test_texts)

    # Train a Classifier
    clf = RidgeClassifier(tol=1e-2, solver="sparse_cg")
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    acc = accuracy_score(y_test, preds)
    acc *= 100
    print(f"Accuracy: {acc:.3f}")

    # Confusion matrix
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, preds, labels=label_names, display_labels=label_names, ax=ax)

    ax.set_title("Confusion Matrix (Custom Document Classifier)")
    plt.show()

def document_classification(file):
    global clf 
    global vectorizer
    global label_names
    
    if clf is None or vectorizer is None:
        train()
    if file.name.endswith(".pdf"):
        doc = fitz.open(file.name)
        content = "\n".join([page.get_text() for page in doc])
        doc.close()
    else:    
        with open(file.name, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    vec = vectorizer.transform([content])
    prediction = clf.predict(vec)[0]
    return f"Document classifyed as a: **{prediction}**. Predictor has an internal accuracy level of **{acc}**%."

train()

demo = gr.Interface(
    fn=document_classification,
    inputs=gr.File(label="Upload .txt or .pdf Document", file_types=[".txt", ".pdf"]),
    outputs=gr.Markdown(label="Prediction"),
    title="Custom Document Classifier",
    description="Upload a text document to classify it as a contract, resume, or invoice."
)

if __name__ == "__main__":
    demo.launch()
