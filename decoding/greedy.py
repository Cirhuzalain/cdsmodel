import torch 

def greedy_search(model, opt, src, fields, max_len=100):
    model.eval()
    batch_size = src.size(0)
    beam_size = opt.beam_size
    device = src.device

    encoder = model.encoder
    decoder = model.decoder
    generator = model.generator

    src = src.repeat(1, beam_size).view(batch_size*beam_size, -1)
    src_pad = src.eq(fields["src"].pad_id)
    src_out = encoder(src, src_pad)

    starts = torch.Tensor([fields['tgt'].vocab.stoi['<bos>']]).long().to(device).expand(batch_size*5, 1).long()

    preds = [starts.view(batch_size, beam_size, -1)[:, 0, :]]
    scores = []

    finish_mask = torch.Tensor([0]*batch_size).byte().to(device)
    xs = starts
    previous = None

    for ts in range(max_len):
        tgt_pad = xs.eq(fields["tgt"].pad_id)
        score, previous = decoder(xs, src_out, src_pad, tgt_pad, previous, ts)
        score = generator(score)

        _scores, _preds = score.max(dim=-1)

        preds.append(_preds.view(batch_size, beam_size, -1)[:, 0, :])
        scores.append(_scores.view(batch_size, beam_size, -1)[:, 0, :].view(-1)*(finish_mask == 0).float())
        finish_mask += (_preds.view(batch_size, beam_size, -1)[:, 0, :] == 2).byte().view(-1)

        if not (torch.any(~finish_mask.bool())):
            break

        xs = _preds

    preds = torch.cat(preds, dim=-1)

    return preds