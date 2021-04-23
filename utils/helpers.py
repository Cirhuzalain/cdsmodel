import os
import re
import subprocess
import tempfile
import random
import datetime
import json

import torch
import numpy as np

from rouge_score import rouge_scorer, scoring
from typing import List, Dict

ROUGE_KEYS = ["rouge1", "rouge2", "rougeL"]
def calculate_rouge(outputs, references):
    scorer = rouge_scorer.RougeScorer(ROUGE_KEYS, use_stemmer=True)
    aggregator = scoring.BootstrapAggregator()

    for reference, output in zip(references, outputs):
        scores = scorer.score(reference, output)
        aggregator.add_scores(scores)

    result = aggregator.aggregate()
    return {k: v.mid.fmeasure*100 for k, v in result.items()}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def printing_opt(opt):
    return "\n".join(["%15s : %s" %(e[0], e[1]) for e in sorted(vars(opt).items(), key=lambda x:x[0])])

class Saver(object):
    """
        Logging ROUGE result and saving model weight during training/validation
    """
    def __init__(self, opt):
        self.ckpt_names = []
        self.max_to_keep = opt.max_to_keep
        self.model_path = opt.model_path + datetime.datetime.now().strftime("-%y%m%d-%H%M%S")
        os.mkdir(self.model_path)

        with open(os.path.join(self.model_path, "params.json"), "w", encoding="UTF-8") as log:
            log.write(json.dumps(vars(opt), indent=4) + "\n")

    def save(self, save_dict, step, rouge, loss):
        filename = "checkpoint-step-%06d" % step
        full_filename = os.path.join(self.model_path, filename)
        self.ckpt_names.append(full_filename)
        torch.save(save_dict, full_filename)

        with open(os.path.join(self.model_path, "log"), "a", encoding="UFT-8") as log:
            log.write("%s\t step : %6d\t loss: %.2f ROUGE 1 : %.2f\t ROUGE 2 : %.2f\t ROUGE L : %.2f\n" % (datetime.datetime.now(), step, loss, rouge["rouge1"], rouge["rouge2"], rouge["rougeL"]))

        if 0 < self.max_to_keep < len(self.ckpt_names):
            earliest_ckpt = self.ckpt_names.pop(0)
            os.remove(earliest_ckpt)
