from neural_network_without_package import neural_network_without_package
from models import download_model, load_model_from_local
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


if __name__ == "__main__":
    # Neural Network without using package
    # neural_network_without_package()

    # Download models from Huggingface
    # model_name : Model name in Huggingface, save_model_path: Saving path in local
    # download_model(save_model_path="sst-3")

    sentiment_pipeline = load_model_from_local("sst-3")
    print(sentiment_pipeline(["I am good", "I am not good"]))
    # main()
