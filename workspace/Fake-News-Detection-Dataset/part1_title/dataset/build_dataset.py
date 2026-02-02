from torch.utils.data import Dataset
import json 
import pandas as pd
import torch
import os
from glob import glob

import logging
from typing import Union

_logger = logging.getLogger('train')

class FakeDataset(Dataset):
    def __init__(self, tokenizer):
        # tokenizer
        self.tokenizer = tokenizer

    def load_dataset(self, data_dir, split, direct_dir: Union[None, str] = None, saved_data_path: bool = False):

        data_info = glob(os.path.join(data_dir, split, '*/*/*'))
        if direct_dir:
            exclude_info = glob(os.path.join(data_dir, split, 'Clickbait_Auto/*/*'))
            include_info = glob(os.path.join(direct_dir, split, '*/*/*'))
            data_info = list(set(data_info) - set(exclude_info))
            data_info = data_info + include_info

        setattr(self, 'saved_data_path', saved_data_path)

        if saved_data_path:
            _logger.info('load saved data')
            data = torch.load(os.path.join(saved_data_path, f'{split}.pt'))
        else:
            _logger.info('load raw data')

            data = {}
            bad_files = 0

            for filename in data_info:
                # 1) 확장자 체크: json 아닌 건 스킵
                if not filename.lower().endswith(".json"):
                    bad_files += 1
                    continue

                # 2) 내용이 ZIP(PK)면 스킵 (확장자가 json이어도 PK면 바이너리)
                try:
                    with open(filename, "rb") as fb:
                        if fb.read(2) == b"PK":
                            bad_files += 1
                            continue
                except Exception:
                    bad_files += 1
                    continue

                # 3) json load: utf-8 기본, 실패하면 cp949로 재시도
                try:
                    with open(filename, "r", encoding="utf-8") as fp:
                        f = json.load(fp)
                except UnicodeDecodeError:
                    # 인코딩 깨지는 파일 구제 (필요없으면 continue로 버려도 됨)
                    try:
                        with open(filename, "r", encoding="cp949", errors="replace") as fp:
                            f = json.load(fp)
                    except Exception:
                        bad_files += 1
                        continue
                except Exception:
                    bad_files += 1
                    continue

                data[filename] = f

            # data_info도 실제로 로드된 파일만 남기기 (중요!)
            data_info = list(data.keys())

            _logger.info(f"Loaded {len(data_info)} files, skipped {bad_files} files")


        setattr(self, 'data_info', data_info)
        setattr(self, 'data', data)

    def transform(self):
        raise NotImplementedError

    def padding(self):
        raise NotImplementedError

    def __getitem__(self, i: int) -> Union[dict, int]:
        if self.saved_data_path:
            doc = {}
            for k in self.data['doc'].keys():
                doc[k] = self.data['doc'][k][i]

            label = self.data['label'][i]

            return doc, label
        
        else:
            news_info = self.data[self.data_info[i]]
        
            # label
            label = 1 if 'NonClickbait_Auto' not in self.data_info[i] else 0
        
            # transform and padding
            doc = self.transform(
                title = news_info['labeledDataInfo']['newTitle'], 
                text  = news_info['sourceDataInfo']['newsContent'].split('\n')
            )

            return doc, label

    def __len__(self):
        raise NotImplementedError
