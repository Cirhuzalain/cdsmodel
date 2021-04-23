import torch 
from decoding.beam import Beam

def generate_beam(min_n_best, model, opt, src, fields, previous_hyps=None, 
                    similarity_metric='hamming',
                    similarity_threshold=0, verbose=False):
    batch_size = src.size(0)
    beam_size = opt.beam_size
    device = src.device 
    model.eval()

    encoder = model.encoder
    decoder = model.decoder
    generator = model.generator
    beams = [Beam(beam_size,
                    min_n_best=min_n_best,
                    eos_token=fields["tgt"].eos_id,
                    padding_token=fields["tgt"].pad_id,
                    bos_token=fields["tgt"].bos_id,
                    device=device,
                    similarity_metric=similarity_metric,
                    similarity_threshold=similarity_threshold) for _ in range(batch_size)]
    
    src = src.repeat(1, beam_size).view(batch_size * beam_size, -1)
    src_pad = src.eq(fields["src"].pad_id)
    src_out = encoder(src, src_pad)

    starts = torch.Tensor([fields["tgt"].vocab.stoi['<bos>']]).long().to(device).expand(batch_size*beam_size, 1).long()

    decoder_input = starts 
    previous = None

    with torch.no_grad():
        for ts in range(opt.max_length):
            if all((b.is_done() for b in beams)):
                break

            tgt_pad = decoder_input.eq(fields["tgt"].pad_id)
            score, previous = decoder(decoder_input, src_out, src_pad, tgt_pad, previous, ts)

            score = score[:, -1, :]
            score = generator(score).view(batch_size, beam_size, -1)

            for i, b in enumerate(beams):
                if not b.is_done():
                    if previous_hyps is None:
                        previous_hyps = [{} for i in range(batch_size)]

                    b.advance(score[i], previous[i])

            selection = torch.cat([b.get_output_from_current_step() for b in beams]).unsqueeze(-1)
            decoder_input = selection

    beam_preds_scores = [list(b.get_top_hyp()) for b in beams]

    return beam_preds_scores, beams