import logging

import torch
import os

from dataloader.utils import load_dataset
from decoding.generation import generate_beam
from decoding.greedy import greedy_search
from architecture.model import CDS
from utils.helpers import get_device, calculate_rouge, set_seed
from utils.cmdopt import parse_sum_args

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
opt = parse_sum_args()
device = get_device()

set_seed(55)

def test(dataset, fields, model):
    
    already, hypothesis, references = 0, [], []

    for batch in dataset:
        if opt.tf:
            predictions = greedy_search(model, opt, batch.src, fields, opt.max_length)
        else:
            predictions, _ = generate_beam(5, model, opt, batch.src, fields)
            predictions = [p for p, _ in predictions]

        hypothesis += [fields["tgt"].decode(p) for p in predictions]
        already += len(predictions)
        logging.info("Summarized: %7d/%7d" % (already, dataset.num_examples))
        references += [fields["tgt"].decode(t) for t in batch.tgt]

    rouge = calculate_rouge(hypothesis, references)
    logging.info("ROUGE 1: %3.2f\tROUGE 2: %3.2f\tROUGE L: %3.2f\t" % (rouge['rouge1'], rouge['rouge2'], rouge['rougeL']))

    with open(opt.output, "w", encoding="UTF-8") as out_file:
        out_file.write("\n".join(hypothesis))
        out_file.write("\n")

    logging.info("Summarization evaluation finished")

def main():
    logging.info("Load dataset...")
    dataset = load_dataset(opt, [opt.input, opt.truth], opt.vocab, device, train=False)

    fields = dataset.fields
    pad_ids = {"src": fields["src"].pad_id, "tgt": fields["tgt"].pad_id}
    vocab_sizes = {"src": len(fields["src"].vocab), "tgt": len(fields["tgt"].vocab)}

    logging.info("Load checkpoint from %s." % opt.model_path)
    checkpoint = torch.load(opt.model_path, map_location=lambda storage, loc: storage)

    logging.info("Load model...")
    model = CDS.load_model(checkpoint["opt"], pad_ids, vocab_sizes, checkpoint["model"]).to(device).eval()

    logging.info("Start summarization...")
    with torch.set_grad_enabled(False):
        test(dataset, fields, model)

if __name__ == '__main__':
    main()