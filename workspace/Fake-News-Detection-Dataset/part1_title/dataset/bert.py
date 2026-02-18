from .build_dataset import FakeDataset
import torch 
from typing import List

class BERTDataset(FakeDataset):
    def __init__(self, tokenizer, max_word_len: int):
        super(BERTDataset, self).__init__(tokenizer=tokenizer)

        self.max_word_len = max_word_len
        # tokenizer에서 vocab 가져오기 (transformers tokenizer 호환)
        if not hasattr(self, "vocab") or self.vocab is None:
            if hasattr(self.tokenizer, "get_vocab"):
                self.vocab = self.tokenizer.get_vocab()
            elif hasattr(self.tokenizer, "vocab"):
                self.vocab = self.tokenizer.vocab
            else:
                self.vocab = {}

        # special token index
        if hasattr(self.tokenizer, "pad_token_id") and self.tokenizer.pad_token_id is not None:
            self.pad_idx = self.tokenizer.pad_token_id
        else:
            self.pad_idx = self.vocab.get("[PAD]", 0) if isinstance(self.vocab, dict) else 0

        if hasattr(self.tokenizer, "cls_token_id") and self.tokenizer.cls_token_id is not None:
            self.cls_idx = self.tokenizer.cls_token_id
        else:
            self.cls_idx = self.vocab.get("[CLS]", 0) if isinstance(self.vocab, dict) else 0


    def transform(self, title: str, text: list) -> dict:
        # text가 문장 리스트면 합치기
        if isinstance(text, (list, tuple)):
            body = " ".join([t for t in text if isinstance(t, str)])
        else:
            body = str(text)
    
        # 제목 + 본문 합치기
        src = f"{title} {body}".strip()
    
        input_ids, token_type_ids, attention_mask = self.tokenize(src)
    
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
        }




    def tokenize(self, src):
        # gluonnlp BERTSPTokenizer는 transformers처럼 truncation/padding kwargs를 받지 않음
        # 1) 토큰화 -> token ids
        token_ids = self.tokenizer(src)
         # ✅ tokenizer가 str 토큰을 내놓는 경우 -> id로 변환
        if len(token_ids) > 0 and isinstance(token_ids[0], str):
            # gluonnlp vocab 객체(Factory에서 만든 BERTVocab)를 tokenizer가 들고 있을 수 있음
            v = getattr(self.tokenizer, "vocab", None) or getattr(self.tokenizer, "_vocab", None)

            # 혹시 없으면(예외 케이스) self.vocab를 시도
            if v is None:
                v = self.vocab

            mapped = []
            for t in token_ids:
                try:
                    mapped.append(int(v[t]))  # BERTVocab는 보통 vocab[token] 가능
                except Exception:
                    mapped.append(0)  # unknown fallback
            token_ids = mapped


        # 2) [CLS], [SEP] 추가
        cls_id = self.vocab.get('[CLS]', 2) if isinstance(self.vocab, dict) else 2
        sep_id = self.vocab.get('[SEP]', 3) if isinstance(self.vocab, dict) else 3
        pad_id = self.vocab.get('[PAD]', 0) if isinstance(self.vocab, dict) else 0

        input_ids = [cls_id] + token_ids + [sep_id]

        # 3) max_len 기준으로 자르기/패딩
        if len(input_ids) > self.max_word_len:
            input_ids = input_ids[:self.max_word_len]
        else:
            input_ids = input_ids + [pad_id] * (self.max_word_len - len(input_ids))

        # 4) token_type_ids / attention_mask 생성 (단일 문장이라 token_type_ids는 0)
        token_type_ids = [0] * self.max_word_len
        attention_mask = [0 if i == pad_id else 1 for i in input_ids]

        return input_ids, token_type_ids, attention_mask





    def length_processing(self, src: list) -> list:
        max_word_len = self.max_word_len - 3 # 3 is the number of special tokens. ex) [CLS], [SEP], [SEP]
        
        cnt = 0
        processed_src = []
        for sent in src:
            cnt += len(sent)
            if cnt > max_word_len:
                sent = sent[:len(sent) - (cnt-max_word_len)]
                processed_src.append(sent)
                break

            else:
                processed_src.append(sent)

        return processed_src


    def pad(self, data: list, pad_idx: int) -> list:
        data = data + [pad_idx] * max(0, (self.max_word_len - len(data)))
        return data


    def padding_bert(self, input_ids: list, token_type_ids: list) -> List[torch.Tensor]:
        # padding using bert models (bts, kobertseg)        
        input_ids = torch.tensor(self.pad(input_ids, self.pad_idx))
        token_type_ids = torch.tensor(self.pad(token_type_ids, self.pad_idx))

        attention_mask = ~(input_ids == self.pad_idx)

        return input_ids, token_type_ids, attention_mask


    def get_token_type_ids(self, input_ids: list) -> list:
        # for segment token
        token_type_ids = []
        for i, v in enumerate(input_ids):
            if i % 2 == 0:
                token_type_ids.append([0] * len(v))
            else:
                token_type_ids.append([1] * len(v))
        return token_type_ids


    def __len__(self):
        if self.saved_data_path:
            return len(self.data['doc']['input_ids'])
        else:
            return len(self.data)
    


