
import os
import json
from typing import Union, Dict
import ast

import pandas as pd
from fastNLP import DataSet, Instance
from fastNLP.io import Loader, DataBundle


class GSM8KLoader(Loader):

    def _load(self, path: str) -> DataSet:
        ds = DataSet()

        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                if line.strip():
                    data = json.loads(line)
                    instance = Instance(**data)
                    ds.append(instance)

        return ds

    def load(self, paths: Union[str, Dict[str, str]] = './data/gsm8k') -> DataBundle:
        if isinstance(paths, str):
            paths = {
                'train': os.path.join(paths, 'train_socratic.jsonl'),
                'dev': os.path.join(paths, 'test_socratic.jsonl'),
                'test': os.path.join(paths, 'test_socratic.jsonl')
            }

        return DataBundle(datasets={k: self._load(v) for k, v in paths.items()})

class GSM8KLoader_kt(Loader):

    def _load(self, path: str) -> DataSet:
        ds = DataSet()

        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                if line.strip():
                    data = json.loads(line)
                    instance = Instance(**data)
                    ds.append(instance)

        return ds

    def load(self, paths: Union[str, Dict[str, str]] = './data/gsm8k') -> DataBundle:
        if isinstance(paths, str):
            paths = {
                'train': os.path.join(paths, 'merged_filtered_single.jsonl'),
                'dev': os.path.join(paths, 'merged_filtered_single.jsonl'),
                'test': os.path.join(paths, 'merged_filtered_single.jsonl')
            }

        return DataBundle(datasets={k: self._load(v) for k, v in paths.items()})
class GSM8KLoader_all_2(Loader):

    def _load(self, path: str) -> DataSet:
        ds = DataSet()

        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                if line.strip():
                    data = json.loads(line)
                    instance = Instance(**data)
                    ds.append(instance)

        return ds

    def load(self, paths: Union[str, Dict[str, str]] = './data/gsm8k') -> DataBundle:
        if isinstance(paths, str):
            paths = {
                'train': os.path.join(paths, 'new_train_all_2.jsonl'),
                'dev': os.path.join(paths, 'new_test_all_2.jsonl'),
                'test': os.path.join(paths, 'new_test_all_2.jsonl')
            }

        return DataBundle(datasets={k: self._load(v) for k, v in paths.items()})
class GSM8KLoader_all_2_2(Loader):

    def _load(self, path: str) -> DataSet:
        ds = DataSet()

        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                if line.strip():
                    data = json.loads(line)
                    instance = Instance(**data)
                    ds.append(instance)

        return ds

    def load(self, paths: Union[str, Dict[str, str]] = './data/gsm8k') -> DataBundle:
        if isinstance(paths, str):
            paths = {
                'train': os.path.join(paths, 'new_train_all_2_2.jsonl'),
                'dev': os.path.join(paths, 'new_test_all_2.jsonl'),
                'test': os.path.join(paths, 'new_test_all_2.jsonl')
            }

        return DataBundle(datasets={k: self._load(v) for k, v in paths.items()})
class GSM8KLoader_all_qwen(Loader):

    def _load(self, path: str) -> DataSet:
        ds = DataSet()

        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                if line.strip():
                    data = json.loads(line)
                    instance = Instance(**data)
                    ds.append(instance)

        return ds

    def load(self, paths: Union[str, Dict[str, str]] = './data/gsm8k') -> DataBundle:
        if isinstance(paths, str):
            paths = {
                'train': os.path.join(paths, 'new_train_all_qwen.jsonl'),
                'dev': os.path.join(paths, 'new_test_all_qwen.jsonl'),
                'test': os.path.join(paths, 'new_test_all_qwen.jsonl')
            }

        return DataBundle(datasets={k: self._load(v) for k, v in paths.items()})
class AQuALoader(Loader):

    def _load(self, path: str) -> DataSet:
        ds = DataSet()
        with open(path) as f:
            ins_list = json.load(f)

        for ins in ins_list:
            instance = Instance(**ins)
            ds.append(instance)

        return ds

    def load(self, paths: Union[str, Dict[str, str]] = './data/AQuA-master') -> DataBundle:
        if isinstance(paths, str):
            paths = {
                'train': os.path.join(paths, 'gsm_style_train.jsonl'),
                'dev': os.path.join(paths, 'gsm_style_dev.jsonl'),
                'test': os.path.join(paths, 'gsm_style_test.jsonl')
            }

        return DataBundle(datasets={k: self._load(v) for k, v in paths.items()})


class DULoader(Loader):

    def _load(self, path: str) -> DataSet:
        ds = DataSet()
        with open(path) as f:
            ins_list = json.load(f)

        for ins in ins_list:
            instance = Instance(**ins)
            ds.append(instance)

        return ds

    def load(self, paths: Union[str, Dict[str, str]] = './data/DU') -> DataBundle:
        if isinstance(paths, str):
            paths = {
                'train': os.path.join(paths, 'date_understanding_gsm_style.json'),
                'dev': os.path.join(paths, 'date_understanding_gsm_style.json'),
                'test': os.path.join(paths, 'date_understanding_gsm_style.json')
            }

        return DataBundle(datasets={k: self._load(v) for k, v in paths.items()})


class StrategyQALoader(GSM8KLoader):


    def load(self, paths: Union[str, Dict[str, str]] = './data/strategyQA_train') -> DataBundle:
        if isinstance(paths, str):
            paths = {
                'train': os.path.join(paths, 'new_train_all_strategyqa.jsonl'),
                'dev': os.path.join(paths, 'new_test_all_strategyqa.jsonl'),
                'test': os.path.join(paths, 'new_test_all_strategyqa.jsonl')
            }
        return DataBundle(datasets={k: self._load(v) for k, v in paths.items()})



class AugASDivLoader(GSM8KLoader):
    def load(self, paths: Union[str, Dict[str, str]] = './data/ASDiv-Aug') -> DataBundle:
        if isinstance(paths, str):
            paths = {
                'train': os.path.join(paths, 'new_train_all_asdiv.jsonl'),
                'dev': os.path.join(paths, 'new_test_all_asdiv.jsonl'),
                'test': os.path.join(paths, 'new_test_all_asdiv.jsonl')
            }

        return DataBundle(datasets={k: self._load(v) for k, v in paths.items()})


class AugASDivLoader_3(GSM8KLoader):
    def load(self, paths: Union[str, Dict[str, str]] = './data/ASDiv-Aug') -> DataBundle:
        if isinstance(paths, str):
            paths = {
                'train': os.path.join(paths, 'new_train_all_asdiv_3.jsonl'),
                'dev': os.path.join(paths, 'new_test_all_asdiv.jsonl'),
                'test': os.path.join(paths, 'new_test_all_asdiv.jsonl')
            }

        return DataBundle(datasets={k: self._load(v) for k, v in paths.items()})
class AugASDivLoader_qwen_3(GSM8KLoader):
    def load(self, paths: Union[str, Dict[str, str]] = './data/ASDiv-Aug') -> DataBundle:
        if isinstance(paths, str):
            paths = {
                'train': os.path.join(paths, 'new_train_all_asdiv_qwen_3.jsonl'),
                'dev': os.path.join(paths, 'new_test_all_aug_qwen.jsonl'),
                'test': os.path.join(paths, 'new_test_all_aug_qwen.jsonl')
            }

        return DataBundle(datasets={k: self._load(v) for k, v in paths.items()})