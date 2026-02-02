from .build_dataset import FakeDataset
import torch 
from typing import List

class BERTDataset(FakeDataset):
    def __init__(self, tokenizer, max_word_len: int):
        super(BERTDataset, self).__init__(tokenizer=tokenizer)

        self.max_word_len = max_word_len

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
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }



    def tokenize(self, src):
        """
        src: transform()에서 넘어오는 문자열(제목+본문 등) 또는 문자열 리스트.
        여기서는 안전하게 문자열로 합쳐서 HF tokenizer로 인코딩한다.
        """
        # src가 리스트(문장 리스트)이면 공백으로 합치기
        if isinstance(src, (list, tuple)):
            text = " ".join([s for s in src if isinstance(s, str)])
        else:
            text = str(src)

        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_word_len,
            return_tensors=None
        )

        input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(encoded["attention_mask"], dtype=torch.long)

        # 일부 tokenizer는 token_type_ids가 없을 수 있음(예: RoBERTa 계열)
        token_type_ids = torch.tensor(
            encoded.get("token_type_ids", [0] * len(encoded["input_ids"])),
            dtype=torch.long
        )

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
    


