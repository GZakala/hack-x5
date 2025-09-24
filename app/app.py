import torch
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
from transformers import AutoModelForTokenClassification, AutoTokenizer

app = FastAPI(title="NER Service")

# словари меток должны совпадать с теми, что были при обучении
label2id = {
    "O": 0, 
    "B-BRAND": 1, "B-TYPE": 2, "B-VOLUME": 3, "B-PERCENT": 4,
    "I-BRAND": 5, "I-TYPE": 6, "I-VOLUME": 7, "I-PERCENT": 8,
}
id2label = {v: k for k, v in label2id.items()}
id2label = {v: k for k, v in label2id.items()}

# создаём "чистую" модель с такой же архитектурой
model = AutoModelForTokenClassification.from_pretrained(
    "cointegrated/rubert-tiny2",
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id,
)

# загружаем веса
state_dict = torch.load("../models/ner_model.pt", map_location="cpu")
model.load_state_dict(state_dict, strict=True)
model.eval()

# загружаем токенайзер
tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")


def predict(text: str):
    text = text.replace("\xa0", " ")
    model.eval()
    enc = tokenizer(text, return_tensors="pt", return_offsets_mapping=True, truncation=True, padding="max_length")
    input_ids = enc["input_ids"].to('cpu')
    attention_mask = enc["attention_mask"].to('cpu')
    offsets = enc["offset_mapping"][0]

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.argmax(dim=-1)[0].cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])

    return logits, offsets, tokens

def decode_predictions(text, offsets, labels):
    tokens = text.split(' ')
    token_offsets = []
    cur_i = 0
    for token in tokens:
        start = cur_i
        end = start + len(token)
        token_offsets.append((start, end))
        cur_i += len(token) + 1

    bio_start_offsets = [int(o[0]) for o in offsets if o[0] != o[1]]
    res = []
    for token, (start, end) in zip(tokens, token_offsets):
        idx_token_label = bio_start_offsets.index(start) + 1
        label = labels[idx_token_label]
        res.append((token, (start, end, id2label[label])))

    return res


# ---------------------------
# Схема запроса
# ---------------------------
class PredictRequest(BaseModel):
    input: str

# ---------------------------
# Схема ответа
# ---------------------------
class EntityResponse(BaseModel):
    start_index: int
    end_index: int
    entity: str

# ---------------------------
# Заглушка для функции предсказания
# ---------------------------
def predict_entities(text: str) -> List[Dict]:
    """
    Здесь должна быть логика NER.
    Возвращает список словарей с ключами: start_index, end_index, entity
    """
    logits, offsets, _ = predict(text)
    token_spans = decode_predictions(text, offsets, logits)
    return [{'start_index': s[0], 'end_index': s[1], 'entity': s[2]} for _, s in token_spans]

# ---------------------------
# Эндпоинт POST /api/predict
# ---------------------------
@app.post("/api/predict", response_model=List[EntityResponse])
async def app_predict(request: PredictRequest):
    text = request.input
    if text.strip() == '':
        return []
    entities = predict_entities(text)
    return entities

# ---------------------------
# Запуск: uvicorn main:app --reload
# ---------------------------
