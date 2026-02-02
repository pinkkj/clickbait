# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import torch
from transformers import BertTokenizer, BertForSequenceClassification

CKPT = "/saved_model/Part1/BERT/best_model.pt"
PRETRAINED = "/workspace/Fake-News-Detection-Dataset/part1_title/.cache/kobert_local"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = BertTokenizer.from_pretrained(PRETRAINED)
model = BertForSequenceClassification.from_pretrained(PRETRAINED, num_labels=2)

state = torch.load(CKPT, map_location="cpu")
model.load_state_dict(state, strict=True)
model.to(device)
model.eval()

def predict(title: str, body: str, max_len=512):
    # 학습처럼 title + body를 한 문자열로 concat
    src = f"{title} {body}".strip()

    inputs = tokenizer(
        src,
        truncation=True,          # single이라 only_second 의미 없음
        padding="max_length",
        max_length=max_len,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    pred = int(probs.argmax())
    return pred, probs



if __name__ == "__main__":
    title = "강타·정유미, 올가을에 결혼?…양측 \\\"결정된 바 없다\\\""

    summary = """
그룹 H.O.T. 멤버 강타와 배우 정유미의 가을 결혼설에 대해 양측 소속사가 \\\"결정된 바가 없다\\\"고 입장을 밝혔다.\n강타의 소속사 SM엔터테인먼트 측은 27일 \\\"두 사람이 좋은 만남을 이어가고 있으며, 아직 결정된 사항은 없다\\\"고 알렸다.\n이날 오전 연예 매체를 중심으로 강타와 정유미가 2년간의 열애 끝에 오는 가을 중 결혼식을 올리며 가까운 친인척들에게 이 사실을 전했다는 보도가 나왔다.\n정유미의 소속사 미스틱스토리 측 역시 \\\"예쁘게 잘 만나고 있지만 가을 결혼에 대해서는 아직 결정된 사안이 없다\\\"고 밝혔다.\n강타와 정유미는 지난 2020년 2월 열애를 인정한 뒤 2년째 공개 연애를 이어오고 있다.\n강타는 1996년 그룹 H.O.T. 멤버로 데뷔한 1세대 아이돌 대표 주자로 솔로 가수 겸 뮤지컬배우 활동을 병행하고 있다.\n정유미는 2004년 KBS 드라마 '애정의 조건'으로 데뷔, SBS '천일의 약속' '옥탑방 왕세자' MBC '검법남녀2' 등에 출연했다.

    """

    pred, probs = predict(title, summary)
    print("pred:", pred, "probs:", probs)

    print("\nNOTE: 0 - nonclickbait / 1 - clickbait")

# -


