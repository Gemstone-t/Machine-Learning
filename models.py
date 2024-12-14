from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import os
import shutil


def download_model(
    model_name="distilbert-base-uncased-finetuned-sst-2-english", save_model_path=""
):

    if save_model_path == "":
        folder_path = "./models/" + model_name
    else:
        folder_path = "./models/" + save_model_path

    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)

    os.makedirs(folder_path)

    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model.save_pretrained(folder_path)
    tokenizer.save_pretrained(folder_path)


def load_model_from_local(load_model_path):

    model_path = "./models/" + load_model_path
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Create a pipeline using the local model and tokenizer
    sentiment_pipeline = pipeline(
        "sentiment-analysis", model=model, tokenizer=tokenizer
    )

    return sentiment_pipeline
