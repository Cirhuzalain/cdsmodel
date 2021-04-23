import logging

import torch
import torch.cuda
from torch.utils.tensorboard import SummaryWriter

from dataloader.utils import load_dataset
from decoding.generation import generate_beam
from optimization.optim import Optimization, LabelSmoothing
from architecture.model import CDS
from utils.helpers import Saver, calculate_rouge, set_seed, get_device, printing_opt
from utils.cmdopt import parse_train_args

writer = SummaryWriter('runs/cds_experiment')
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
opt = parse_train_args()

device = get_device()
rouge = None

logging.info("\n" + printing_opt(opt))

saver = Saver(opt)

set_seed(55)


def valid(model, criterion, valid_dataset, step):

    hypothesis, references = [], []

    for batch in valid_dataset:
        scores = model(batch.src, batch.tgt)
        loss = criterion(scores, batch.tgt)
        total_loss += loss.data
        total += 1

        if opt.tf:
            _, predictions = scores.topk(k=1, dim=-1)
        else:
            predictions = generate_beam(5, model, opt, batch.src, valid_dataset.fields)

        hypothesis += [valid_dataset.fields["tgt"].decode(p) for p in predictions]
        references += [valid_dataset.fields["tgt"].decode(t) for t in batch.tgt]

    rouge = calculate_rouge(hypothesis, references)

    writer.add_scalar("ROUGE 1", rouge["rouge1"], step)
    writer.add_scalar("ROUGE 2", rouge["rouge2"], step)
    writer.add_scalar("ROUGE L", rouge["rougeL"], step)
    writer.add_scalar("Validation loss", total_loss / total, step)
    logging.info("Valid loss : %.2f\t Valid ROUGE 1 : %3.2f \t Valid ROUGE 2: %3.2f \t Valid ROUGE L: %3.2f" % (total_loss / total, rouge['rouge1'], rouge['rouge2'], rouge['rougeL']))
    checkpoint = {"model" : model.state_dict(), "opt": opt}
    saver.save(checkpoint, step, rouge, total_loss / total)
    return rouge

def train(model, criterion, optimizer, train_dataset, valid_dataset):
    total_loss = 0.0 
    rouge = {"rouge1" : 0.0, "rouge2": 0.0, "rougeL": 0.0}
    model.zero_grad()

    for epoch in range(50):
        for i, batch in enumerate(train_dataset):
            scores = model(batch.src, batch.tgt)
            loss = criterion(scores, batch.tgt)

            loss.backward()
            total_loss += loss.data

            optimizer.step()
            model.zero_grad()

            if optimizer.n_step % opt.report_every == 0:
                mean_loss = total_loss / opt.report_every / opt.grad_accum
                logging.info("Epoch : %7d\t step : %7d\t loss: %7f" % (epoch, optimizer.n_step, mean_loss))
                writer.add_scalar('Training loss', mean_loss, optimizer.n_step)
                total_loss = 0.0

            if optimizer.n_step % opt.save_every == 0:
                with torch.set_grad_enabled(False):
                    rouge = valid(model, criterion, valid_dataset, optimizer.n_step)
                model.train()
            
            del loss
        optimizer.scheduler.step(rouge['rouge2'])

def main():
    logging.info("Load dataset...")
    train_dataset = load_dataset(opt, opt.train, opt.vocab, device, train=True)
    valid_dataset = load_dataset(opt, opt.valid, opt.vocab, device, train=False)
    fields = valid_dataset.fields = train_dataset.fields

    logging.info("Load model...")

    pad_ids = {"src": fields["src"].pad_id, "tgt": fields["tgt"].pad_id}
    vocab_sizes = {"src": len(fields["src"].vocab), "tgt": len(fields["tgt"].vocab)}

    opt.dropout = 0.2668093875953269
    opt.label_smoothing = 0.100411344590318275
    opt.lr = 9.519866129601932e-05

    model = CDS.load_model(opt, pad_ids, vocab_sizes).to(device)
    criterion = LabelSmoothing(opt.label_smoothing, vocab_sizes["tgt"], pad_ids["tgt"]).to(device)

    n_step = int(opt.train_from.split("-")[-1]) if opt.train_from else 1
    optimizer = Optimization(model.parameters(), opt.lr, opt.hidden_size, opt.warm_up, n_step)

    logging.info("Start training...")
    train(model, criterion, optimizer, train_dataset, valid_dataset)
    writer.close()

if __name__ == '__main__':
    main()