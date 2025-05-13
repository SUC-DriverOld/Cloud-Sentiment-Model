import torch
import os
import argparse
from transformers import BertTokenizer, BertModel
from model import CloudSentimentModel


def extract_bert_features(config, text, tokenizer, bert_model, max_length=128, device="cuda"):
    bert_model.eval()
    with torch.no_grad():
        encoded = tokenizer(
            text,
            max_length=max_length,
            padding=config["preprocess"]["padding"],
            truncation=config["preprocess"]["truncation"],
            return_tensors=config["preprocess"]["return_tensors"],
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        features = bert_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
    return features


def predict_single(model: CloudSentimentModel, features) -> bool:
    model.eval()
    with torch.no_grad():
        logits, ex, en, he = model(features)
        pred = torch.argmax(logits, dim=1).item()
        # 计算不确定性指标
        uncertainty = en.item() + he.item()
    return pred, uncertainty


def load_models(model_path) -> tuple:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading tokenizer and model...")
    assert os.path.exists("pretrain/chinese-roberta-wwm-ext-large/pytorch_model.bin"), "Please download the pre-trained model first."

    tokenizer = BertTokenizer.from_pretrained("pretrain/chinese-roberta-wwm-ext-large")
    bert_model = BertModel.from_pretrained("pretrain/chinese-roberta-wwm-ext-large")
    bert_model = bert_model.to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    state_dict = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}
    hidden_size = config["model"]["hidden_size"]

    model = CloudSentimentModel(
        input_size=hidden_size,
        cloud_drop_num=config.model.cloud_drop_num,
        cloud_dim=config.model.get("cloud_dim", 1),
        features=config.model.features,
        dropout=config.model.dropout
    )
    model.load_state_dict(state_dict)

    return model, tokenizer, bert_model, config, device


def predict(model, tokenizer, bert_model, config, text, max_length=128, device="cuda"):
    features = extract_bert_features(config, text, tokenizer, bert_model, max_length, device)
    pred, uncertainty = predict_single(model, features)
    return pred, uncertainty


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="Path to the model checkpoint")
    parser.add_argument("-t", "--text", type=str, default=None, help="Text to predict sentiment")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum length of the input text")
    args = parser.parse_args()

    assert os.path.exists(args.model), f"Model checkpoint not found: {args.model}"

    if args.text:
        model, tokenizer, bert_model, config, device = load_models(args.model)
        model.to(device)
        pred, uncertainty = predict(model, tokenizer, bert_model, config, args.text, args.max_length, device)

        print(f"[Input]: {args.text}")
        print(f"[Prediction]: {'Positive' if pred else 'Negative'}")
        print(f"[Uncertainty]: {uncertainty}")
    else:
        try:
            model, tokenizer, bert_model, config, device = load_models(args.model)
            model.to(device)
            while True:
                text = input(">>> ")
                pred, uncertainty = predict(model, tokenizer, bert_model, config, text, args.max_length, device)
                print(f"Prediction: {'Positive' if pred else 'Negative'}, Uncertainty: {uncertainty}")
        except KeyboardInterrupt:
            os._exit(0)