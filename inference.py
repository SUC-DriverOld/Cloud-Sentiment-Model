import torch
import os
import argparse
from transformers import BertTokenizer, BertModel
from model import CloudSentimentModel


def predict_single(model: CloudSentimentModel, config, tokenizer, text: str, device: str) -> bool:
    model.eval()
    inputs = tokenizer(
        text,
        padding=config.data.padding,
        truncation=config.data.truncation,
        max_length=config.data.max_length,
        return_tensors=config.data.return_tensors
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits, _, _, _ = model(inputs["input_ids"], inputs["attention_mask"])
        pred = torch.argmax(logits, dim=1).item()
    return pred


def inference(model_path: str, text: str) -> bool:
    tokenizer = BertTokenizer.from_pretrained("pretrain/chinese-roberta-wwm-ext-large")
    bert_model = BertModel.from_pretrained("pretrain/chinese-roberta-wwm-ext-large")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    state_dict = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}

    model = CloudSentimentModel(
        bert_model,
        cloud_drop_num=config.model.cloud_drop_num,
        features=config.model.features,
        dropout=config.model.dropout
    )

    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    return predict_single(model, config, tokenizer, text, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, default=None, help="Path to the checkpoint model")
    parser.add_argument("text", type=str, default=None, help="Text to predict")
    args = parser.parse_args()

    print(f"[Input ]: {args.text}")
    pred = inference(args.model, args.text)
    print(f"[Output]: {'positive' if pred else 'negative'}")