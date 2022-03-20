# -*- coding:utf-8 -*-
from processor import Processor
from net import *

import json
from transformers import BertConfig, BertModel
import torch

class Predictor:
    def __init__(self, ckpt_path='./models/saved/final_best.pt'):
        self.device = 'cuda'
        self.model = torch.load(ckpt_path)
        self.model.to(self.device)
        self.model.eval()
        self.processor = Processor(self.device)

    def predict(self, content: dict) -> dict:
        data = self.processor.preprocess(content)
        with torch.no_grad():
            model_out = self.model(**data)
        argument_info = self.processor.extract_entity(model_out, content)
        out = self.processor.postprocess(argument_info, content)
        return out # out: dict

if __name__ == "__main__":
    example_input = '{"text":"08年4月，郑煤集团拟以非公开发行的方式进行煤炭业务整体上市，解决与郑州煤电同业竞争问题，但之后由于股市的大幅下跌导致股价跌破发行价而被迫取消整体上市。","qas":[[{"question":"中心词","answers":[{"start":57,"end":58,"text":"导致"}]}]]}'
    example_output = '{"text":"08年4月，郑煤集团拟以非公开发行的方式进行煤炭业务整体上市，解决与郑州煤电同业竞争问题，但之后由于股市的大幅下跌导致股价跌破发行价而被迫取消整体上市。","qas":[[{"question":"原因中的核心名词","answers":[{"start":50,"end":51,"text":"股市"}]},{"question":"原因中的谓语或状态","answers":[{"start":53,"end":56,"text":"大幅下跌"}]},{"question":"中心词","answers":[{"start":57,"end":58,"text":"导致"}]},{"question":"结果中的核心名词","answers":[{"start":59,"end":60,"text":"股价"}]},{"question":"结果中的谓语或状态","answers":[{"start":61,"end":65,"text":"跌破发行价"}]}]]}'
    obj = Predictor('./models/saved/CBCP_best.pt')
    output = obj.predict(json.loads(example_input))
    print(output == json.loads(example_output))