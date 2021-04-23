import torch

import math
import editdistance
from operator import attrgetter


class HypothesisTail(object):

    slots = ['timestamp', 'hybrid', 'score', 'tokenid']

    def __init__(self, timestep, hypid, score, tokenid):
        self.timestep = timestep
        self.hypid = hypid
        self.score = score
        self.tokenid = tokenid

class Beam(object):
    """
        Beam Search Implementation for AMMI NLP 2 course
    """
    def __init__(self, beam_size, 
                    padding_token=0, 
                    bos_token=1, 
                    eos_token=2, 
                    min_length=3, 
                    min_n_best=3, 
                    device='cpu', 
                    similarity_metric='hamming',
                    similarity_threshold=0):
        self.beam_size = beam_size
        self.min_length = min_length
        self.eos = eos_token
        self.bos = bos_token
        self.pad = padding_token
        self.device = device
        self.scores = None
        self.all_scores = [torch.Tensor([0.0] * beam_size).to(self.device)]
        self.bookkeep = []
        self.outputs = [
            torch.Tensor(self.beam_size).long().fill_(self.bos).to(self.device)
        ]
        self.finished = []
        self.eos_top = False
        self.eos_top_ts = None
        self.n_best_counter = 0
        self.min_n_best = min_n_best
        self.partial_hyps = [[self.bos] for i in range(beam_size)]

        self.history_hyps = []
        self.similarity_metric = similarity_metric
        self.similarity_threshold  = similarity_threshold
        self.banned_tokens = set()

    def get_output_from_current_step(self):
        return self.outputs[-1]

    def get_backtrack_from_current_step(self):
        return self.bookkeep[-1]

    def hamming_distance(self, t1, t2):
        dist = 0
        for tok1, tok2 in zip(t1, t2):
            if tok1 != tok2:
                dist += 1
        return dist

    def edit_distance(self, t1, t2):
        dist = editdistance.eval(t1, t2)
        return dist

    def similarity_check(self, active_hyp, previous_hyps, metric='hamming', threshold=0):
        banned_tokens = []
        active_len = len(active_hyp)

        for observed_hyp, _banned_tokens in previous_hyps.items():
            if len(observed_hyp) != active_len:
                continue
            if metric == 'hamming':
                dist = self.hamming_distance(observed_hyp, active_hyp)
            if metric == 'edit':
                dist = self.edit_distance(observed_hyp, active_hyp)
            if dist <= threshold:
                banned_tokens.extend(_banned_tokens)

        return list(set(banned_tokens))

    def select_paths(self, logprobs, prior_scores, previous_hyps):
        beam_scores = logprobs + prior_scores.unsqueeze(1).expand_as(logprobs)

        current_length = len(self.all_scores)
        if len(previous_hyps) > 0 and current_length > 0:
            for hyp_id in range(beam_scores.size(0)):
                active_hyp = tuple(self.partial_hyps[hyp_id])
                banned_tokens = self.similarity_check(active_hyp, previous_hyps, metric=self.similarity_metric, threshold=self.similarity_threshold)

                if len(banned_tokens) > 0:
                    beam_scores[:, banned_tokens] = -10e5

        flat_beam_scores = beam_scores.view(-1)
        best_scores, best_idxs = torch.topk(flat_beam_scores, self.beam_size, dim=-1)
        voc_size = logprobs.size(-1)

        hyp_ids = best_idxs // voc_size
        tok_ids = best_idxs % voc_size

        return (hyp_ids, tok_ids, best_scores)

    def advance(self, logprobs, previous_hyps):
        current_length = len(self.all_scores) - 1
        if current_length < self.min_length:
            for hyp_id in range(logprobs.size(0)):
                logprobs[hyp_id][self.eos] = -10e5

        if self.scores is None:
            logprobs = logprobs[0:1]
            self.scores = torch.zeros(1).type_as(logprobs).to(logprobs.device)

        hyp_ids, tok_ids, self.scores = self.select_paths(logprobs, self.scores, previous_hyps)
        self.all_scores.append(self.scores.clone())

        self.outputs.append(tok_ids)
        self.bookkeep.append(hyp_ids)
        self.partial_hyps = [
            self.partial_hyps[hyp_ids[i]] + [tok_ids[i].item()]
            for i in range(self.beam_size)
        ]
        self.history_hyps.extend(self.partial_hyps)

        for hypid in range(self.beam_size):
            if self.outputs[-1][hypid] == self.eos:
                self.scores[hypid] = -10e5
                eostail = HypothesisTail(
                    timestep=len(self.outputs) - 1,
                    hypid=hypid,
                    score=self.all_scores[-1][hypid],
                    tokenid=self.eos
                )
                self.finished.append(eostail)
                self.n_best_counter += 1
        
        if self.outputs[-1][0] == self.eos:
            self.eos_top = True
            if self.eos_top_ts is None:
                self.eos_top_ts = len(self.outputs) -1

    def is_done(self):
        return self.eos_top and self.n_best_counter >= self.min_n_best

    def get_top_hyp(self):
        return self.get_rescored_finished(n_best=1)[0]

    def get_hyp_from_finished(self, hypothesis_tail):
        hyp_idx = []
        endback = hypothesis_tail.hypid
        for i in range(hypothesis_tail.timestep, -1, -1):
            hypothesis_tail.append(
                HypothesisTail(
                    timestep=i,
                    hypid=endback,
                    score=self.all_scores[i][endback],
                    tokenid=self.outputs[i][endback]
                )
            )
            endback = self.bookkeep[i - 1][endback]
        
        return hyp_idx

    def get_pretty_hypothesis(self, list_of_hypotails):
        return torch.stack([ht.tokenid for ht in reversed(list_of_hypotails)])

    def get_rescored_finished(self, n_best=None, add_length_penalty=False):
        if not self.finished:
            self.finished.append(
                HypothesisTail(
                    timestep=len(self.outputs) - 1,
                    hypid=0,
                    score=self.all_scores[-1][0],
                    tokenid=self.eos
                )
            )

        rescored_finished = []
        for finished_item in self.finished:
            if add_length_penalty:
                current_length = finished_item.timestep + 1
                length_penalty = math.pow((1 + current_length) / 6, 0.65)
            else:
                length_penalty = 1
            rescored_finished.append(
                HypothesisTail(
                    timestep=finished_item.timestep,
                    hypid=finished_item.hypid,
                    score=finished_item.score / length_penalty,
                    tokenid=finished_item.tokenid
                )
            )

        srted = sorted(rescored_finished, key=attrgetter('score'), reverse=True)

        if n_best is not None:
            srted = srted[:n_best]

        return [
            (self.get_pretty_hypothesis(self.get_hyp_from_finished(hyp)), hyp.score)
            for hyp in srted
        ]




