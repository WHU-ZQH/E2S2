#opyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import random
import numpy as np
import torch
import copy
from . import FairseqDataset, data_utils


def collate(
    samples,
    pad_idx,
    eos_idx,
    vocab,
    left_pad_source=False,
    left_pad_target=False,
    input_feeding=True,
    pad_to_length=None,
):
    assert input_feeding
    if len(samples) == 0:
        return {}

    #print('samples---',samples[0].keys())

    def merge(key, left_pad, move_eos_to_beginning=False, pad_to_length=None):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx=None,  # use eos_idx of each sample instead of vocab.eos()
            left_pad=left_pad,
            move_eos_to_beginning=move_eos_to_beginning,
            pad_to_length=pad_to_length,
        )

    id = torch.LongTensor([s["id"] for s in samples])
    src_tokens = merge(
        "source",
        left_pad=left_pad_source,
        pad_to_length=pad_to_length["source"] if pad_to_length is not None else None,
    )
    # sort by descending source length
    src_lengths = torch.LongTensor([s["source"].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    if samples[0].get("target", None) is not None:
        target = merge(
            "target",
            left_pad=left_pad_target,
            pad_to_length=pad_to_length["target"]
            if pad_to_length is not None
            else None,
        )
        target = target.index_select(0, sort_order)
        ntokens = sum(len(s["target"]) for s in samples)

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                "target",
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
                pad_to_length=pad_to_length["target"]
                if pad_to_length is not None
                else None,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
    else:
        ntokens = sum(len(s["source"]) for s in samples)

    wso_labels =None
    wso_labels = merge("wso_labels",left_pad=left_pad_target,pad_to_length=pad_to_length["wso_labels"] if pad_to_length is not None else None,)
    wso_masks =None
    wso_masks = merge("wso_masks",left_pad=left_pad_target,pad_to_length=pad_to_length["wso_masks"] if pad_to_length is not None else None,)
    wso_masks = wso_masks.index_select(0, sort_order)
    wso_labels = wso_labels.index_select(0, sort_order)
    batch = {
        "id": id,
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
        },
        "target": target,
        'wso_labels':wso_labels,
        'wso_masks':wso_masks,
        "nsentences": samples[0]["source"].size(0),
        "sort_order": sort_order,
    }
    if prev_output_tokens is not None:
        batch["net_input"]["prev_output_tokens"] = prev_output_tokens

    return batch


class DenoisingDataset(FairseqDataset):
    """
    A wrapper around TokenBlockDataset for BART dataset.

    Args:
        dataset (TokenBlockDataset): dataset to wrap
        sizes (List[int]): sentence lengths
        vocab (~fairseq.data.Dictionary): vocabulary
        mask_idx (int): dictionary index used for masked token
        mask_whole_words: only mask whole words. This should be a byte mask
            over vocab indices, indicating whether it is the beginning of a
            word. We will extend any mask to encompass the whole word.
        shuffle (bool, optional): shuffle the elements before batching.
          Default: ``True``
        seed: Seed for random number generator for reproducibility.
        args: argparse arguments.
    """

    def __init__(
        self,
        dataset,
        sizes,
        vocab,
        mask_idx,
        mask_whole_words,
        shuffle,
        seed,
        args,
        eos=None,
        item_transform_func=None,
    ):
        self.dataset = dataset

        self.sizes = sizes

        self.vocab = vocab
        self.shuffle = shuffle
        self.seed = seed
        self.mask_idx = mask_idx
        self.mask_whole_word = mask_whole_words
        self.mask_ratio = args.mask
        self.random_ratio = args.mask_random
        self.insert_ratio = args.insert
        self.rotate_ratio = args.rotate
        self.shuffle_word_ratio = 0.05
        self.permute_sentence_ratio = args.permute_sentences
        self.eos = eos if eos is not None else vocab.eos()
        self.item_transform_func = item_transform_func

        if args.bpe != "gpt2":
            self.full_stop_index = self.vocab.eos()
        else:
            assert args.bpe == "gpt2"
            self.full_stop_index = self.vocab.index("13")

        self.replace_length = args.replace_length
        if self.replace_length not in [-1, 0, 1]:
            raise ValueError(f"invalid arg: replace_length={self.replace_length}")
        if args.mask_length not in ["subword", "word", "span-poisson"]:
            raise ValueError(f"invalid arg: mask-length={args.mask_length}")
        if args.mask_length == "subword" and args.replace_length not in [0, 1]:
            raise ValueError(f"if using subwords, use replace-length=1 or 0")

        self.mask_span_distribution = None
        if args.mask_length == "span-poisson":
            _lambda = args.poisson_lambda

            lambda_to_the_k = 1
            e_to_the_minus_lambda = math.exp(-_lambda)
            k_factorial = 1
            ps = []
            for k in range(0, 128):
                ps.append(e_to_the_minus_lambda * lambda_to_the_k / k_factorial)
                lambda_to_the_k *= _lambda
                k_factorial *= k + 1
                if ps[-1] < 0.0000001:
                    break
            ps = torch.FloatTensor(ps)
            self.mask_span_distribution = torch.distributions.Categorical(ps)

        self.epoch = 0

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return True  # only the noise changes, not item sizes

    def set_epoch(self, epoch, **unused):
        self.epoch = epoch

    def __getitem__(self, index):
        with data_utils.numpy_seed(self.seed, self.epoch, index):
            tokens = self.dataset[index]
            assert tokens[-1] == self.eos
            source, target = tokens, tokens.clone()
            #print('source--{}'.format(source))
            
            #wso_mask=torch.tensor([0]*len(source),dtype=torch.int)

            if self.shuffle_word_ratio > 0.0:
                source, wso_label, wso_mask = self.add_shuffle(source, self.shuffle_word_ratio)
                #print('shuffle--source--{}, idx--{}, mask--{}'.format(source, wso_label, wso_mask))

            if self.permute_sentence_ratio > 0.0:
                source, wso_mask = self.permute_sentences(source, self.permute_sentence_ratio, wso_mask)
                #print('permute--source--{}, idx--{}'.format(source, wso_mask))

            if self.mask_ratio > 0:
                source, wso_mask = self.add_whole_word_mask(source, self.mask_ratio, wso_mask)
                #print('mask--source--{}, idx--{}'.format(source, wso_mask))

            if self.insert_ratio > 0:
                source, wso_mask= self.add_insertion_noise(source, self.insert_ratio, wso_mask)
                #print('insert--source--{}'.format(source))
                #assert 0

            if self.rotate_ratio > 0.0 and np.random.random() < self.rotate_ratio:
                source, wso_mask = self.add_rolling_noise(source, wso_mask)

            #if self.shuffle_word_ratio > 0.0:
                #source, wso_label, _ = self.add_shuffle(source, self.shuffle_word_ratio, wso_mask)
                #del wso_mask

            #print('{}--source--{}'.format(wso_label,wso_mask))
        # there can additional changes to make:
        if self.item_transform_func is not None:
            source, target = self.item_transform_func(source, target)

        assert (source >= 0).all()
        assert (source[1:-1] >= 1).all()
        assert (source <= len(self.vocab)).all()
        assert source[0] == self.vocab.bos()
        assert source[-1] == self.eos
        #assert len(wso_label.nonzero())==len(wso_mask.nonzero())
        return {
            "id": index,
            "source": source,
            "target": target,
            'wso_labels':wso_label,
            'wso_masks':wso_mask
        }

    def __len__(self):
        return len(self.dataset)

    def permute_sentences(self, source, p=1.0, wso_mask=None):
        full_stops = source == self.full_stop_index
        # Pretend it ends with a full stop so last span is a sentence
        full_stops[-2] = 1

        # Tokens that are full stops, where the previous token is not
        sentence_ends = (full_stops[1:] * ~full_stops[:-1]).nonzero(as_tuple=False) + 2
        result = source.clone()
        result_idx = wso_mask.clone()

        num_sentences = sentence_ends.size(0)
        num_to_permute = math.ceil((num_sentences * 2 * p) / 2.0)
        substitutions = torch.randperm(num_sentences)[:num_to_permute]
        ordering = torch.arange(0, num_sentences)
        ordering[substitutions] = substitutions[torch.randperm(num_to_permute)]

        # Ignore <bos> at start
        index = 1
        for i in ordering:
            sentence = source[(sentence_ends[i - 1] if i > 0 else 1) : sentence_ends[i]]
            sentence_idx = wso_mask[(sentence_ends[i - 1] if i > 0 else 1) : sentence_ends[i]]
            result[index : index + sentence.size(0)] = sentence
            result_idx[index : index + sentence.size(0)] = sentence_idx
            index += sentence.size(0)
        return result, result_idx

    def word_starts(self, source):
        if self.mask_whole_word is not None:
            is_word_start = self.mask_whole_word.gather(0, source)
        else:
            is_word_start = torch.ones(source.size())
        is_word_start[0] = 0
        is_word_start[-1] = 0
        return is_word_start

    def add_whole_word_mask(self, source, p, wso_mask=None):
        is_word_start = self.word_starts(source)
        num_to_mask = int(math.ceil(is_word_start.float().sum() * p))
        num_inserts = 0
        if num_to_mask == 0:
            return source, wso_mask

        if self.mask_span_distribution is not None:
            lengths = self.mask_span_distribution.sample(sample_shape=(num_to_mask,))

            # Make sure we have enough to mask
            cum_length = torch.cumsum(lengths, 0)
            while cum_length[-1] < num_to_mask:
                lengths = torch.cat(
                    [
                        lengths,
                        self.mask_span_distribution.sample(sample_shape=(num_to_mask,)),
                    ],
                    dim=0,
                )
                cum_length = torch.cumsum(lengths, 0)

            # Trim to masking budget
            i = 0
            while cum_length[i] < num_to_mask:
                i += 1
            lengths[i] = num_to_mask - (0 if i == 0 else cum_length[i - 1])
            num_to_mask = i + 1
            lengths = lengths[:num_to_mask]

            # Handle 0-length mask (inserts) separately
            lengths = lengths[lengths > 0]
            num_inserts = num_to_mask - lengths.size(0)
            num_to_mask -= num_inserts
            if num_to_mask == 0:
                return self.add_insertion_noise(source, num_inserts / source.size(0), wso_mask)

            assert (lengths > 0).all()
        else:
            lengths = torch.ones((num_to_mask,)).long()
        assert is_word_start[-1] == 0
        word_starts = is_word_start.nonzero(as_tuple=False)
        indices = word_starts[
            torch.randperm(word_starts.size(0))[:num_to_mask]
        ].squeeze(1)
        try :
            while torch.sum(wso_mask[indices]) == 0:
                indices= word_starts[torch.randperm(word_starts.size(0))[:num_to_mask]].squeeze(1)
        except:
            #print('oooo, add_mask')
            return tokens, wso_mask
        mask_random = torch.FloatTensor(num_to_mask).uniform_() < self.random_ratio

        source_length = source.size(0)
        assert source_length - 1 not in indices
        to_keep = torch.ones(source_length, dtype=torch.bool)
        is_word_start[
            -1
        ] = 255  # acts as a long length, so spans don't go over the end of doc
        if self.replace_length == 0:
            if torch.sum(wso_mask[indices])==0:
               to_keep[indices] = 0
        else:
            # keep index, but replace it with [MASK]
            if torch.sum(wso_mask[indices])==0:
               source[indices] = self.mask_idx
               #wso_mask[indices]= -1
            if torch.sum(wso_mask[indices[mask_random]]) == 0:
               source[indices[mask_random]] = torch.randint(1, len(self.vocab), size=(mask_random.sum(),))
               #wso_mask[indices[mask_random]] = -1

        if self.mask_span_distribution is not None:
            assert len(lengths.size()) == 1
            assert lengths.size() == indices.size()
            lengths -= 1
            while indices.size(0) > 0:
                assert lengths.size() == indices.size()
                lengths -= is_word_start[indices + 1].long()
                uncompleted = lengths >= 0
                indices = indices[uncompleted] + 1
                mask_random = mask_random[uncompleted]
                lengths = lengths[uncompleted]
                if self.replace_length != -1:
                    # delete token
                    if torch.sum(wso_mask[indices])==0:
                       to_keep[indices] = 0
                else:
                    # keep index, but replace it with [MASK]
                    if torch.sum(wso_mask[indices])==0:
                       source[indices] = self.mask_idx
                       #wso_mask[indices] = -1
                    if torch.sum(wso_mask[indices[mask_random]]) == 0:
                       source[indices[mask_random]] = torch.randint(1, len(self.vocab), size=(mask_random.sum(),))
                       #wso_mask[indices[mask_random]] = -1
        else:
            # A bit faster when all lengths are 1
            while indices.size(0) > 0:
                uncompleted = is_word_start[indices + 1] == 0
                indices = indices[uncompleted] + 1
                mask_random = mask_random[uncompleted]
                if self.replace_length != -1:
                    # delete token
                    if torch.sum(wso_mask[indices])==0:
                       to_keep[indices] = 0
                else:
                    # keep index, but replace it with [MASK]
                    if torch.sum(wso_mask[indices])==0:  # torch.sum(wso_mask[indices[mask_random]]==0:
                       source[indices] = self.mask_idx
                       #wso_mask[indices] = -1
                    if torch.sum(wso_mask[indices[mask_random]])==0:
                       source[indices[mask_random]] = torch.randint(1, len(self.vocab), size=(mask_random.sum(),))
                       #wso_mask[indices[mask_random]] = -1

                assert source_length - 1 not in indices

        source = source[to_keep]
        wso_mask=wso_mask[to_keep]

        if num_inserts > 0:
            source, wso_mask = self.add_insertion_noise(source, num_inserts / source.size(0), wso_mask)

        return source, wso_mask


    def add_shuffle(self, tokens, p, wso_mask=None):
        """
        Word Structural objective
        """
        is_word_start = self.word_starts(tokens)
        special_tokens = [self.vocab.eos(), self.vocab.bos()]
        indices= [i for i in range(list(tokens.size())[0]) if tokens[i] not in special_tokens]
        pvals = 1. / np.arange(1, 4)  # [1,0.5,0.33]
        pvals /= pvals.sum(keepdims=True)  # [0.545, 0.273, 0.182]
        unigrams = []
        max_seq_len, mask_lm_prob=512, 0.15
        shuffle_window = int(1/(mask_lm_prob))*3
        max_preds_per_seq=math.ceil(max_seq_len * mask_lm_prob / 10) * 10
        for id in indices:
            if len(unigrams) >= 1 and is_word_start[id] == 0:
                unigrams[-1].append(id)
            else:
                unigrams.append([id])
        num_to_predict = min(int(max_preds_per_seq / 3), max(1, int(round(list(tokens.size())[0] * (mask_lm_prob / 3)))))
        offset = 0
        mask_grams = np.array([False] * len(unigrams))
        while offset < len(unigrams):
            n = 1
            ctx_size = min(n * shuffle_window, len(unigrams) - offset)
            m = random.randint(0, ctx_size - 1)
            s = offset + m
            e = min(offset + m + n, len(unigrams))
            offset = max(offset + ctx_size, e)
            if len(unigrams[s]) == 1:
                mask_grams[s:e] = True
        target_labels = [None] * list(tokens.size())[0]
        w_cnt = 0
        for m, word in zip(mask_grams, unigrams):
            if m:
                for idx in word: #(torch.sum(wso_mask[idx-1:idx+2])==0) and 
                    if (tokens[idx] not in special_tokens) and (
                            tokens[idx + 1] not in special_tokens) and (tokens[idx - 1] not in special_tokens):
                        label = self._shuffle(idx, tokens)
                        target_labels[idx - 1:idx + 2] = label
                        w_cnt += 1
                if w_cnt >= num_to_predict:
                    break
        target_labels = [x if x else 0 for x in target_labels]
        wso_mask= [2 if x else 0 for x in target_labels]
        
        #target_labels = torch.tensor(target_labels, dtype=torch.int)
        target_labels = torch.tensor(target_labels, dtype=torch.int)
        wso_mask= torch.tensor(wso_mask, dtype=torch.int)
        return tokens, target_labels, wso_mask

    def _shuffle(self, idx, tokens):
        label = tokens[idx - 1:idx + 2]
        #print('label', label)
        new_split = copy.deepcopy(label)
        indexs=torch.randperm(3)
        new_split=new_split[indexs]
        #print('suffle-label', new_split)
        tokens[idx - 1:idx + 2] = new_split
        return label

    # def add_shuffle(self, source, p):
    #     num_words=len(source)
    #     is_word_start = self.word_starts(source)
    #     num_to_shuffle = int(math.ceil(is_word_start.float().sum() * p))
    #     num_inserts = 0
    #     if num_to_shuffle == 0:
    #         return source
    #
    #     new_split = copy.deepcopy(tokens[idx-1:idx+2])
    #     random.shuffle(new_split)
    #     tokens[idx - 1:idx + 2] = new_split
    #     return tokens


    def add_permuted_noise(self, tokens, p, source_idx=None):
        num_words = len(tokens)
        num_to_permute = math.ceil(((num_words * 2) * p) / 2.0)
        substitutions = torch.randperm(num_words - 2)[:num_to_permute] + 1
        tokens[substitutions] = tokens[substitutions[torch.randperm(num_to_permute)]]
        return tokens

    def add_rolling_noise(self, tokens, source_idx=None):
        offset = np.random.randint(1, max(1, tokens.size(-1) - 1) + 1)
        tokens = torch.cat(
            (tokens[0:1], tokens[offset:-1], tokens[1:offset], tokens[-1:]),
            dim=0,
        )
        source_idx = torch.cat(
            (source_idx[0:1], source_idx[offset:-1], source_idx[1:offset], source_idx[-1:]),dim=0)
        return tokens, source_idx

    def add_insertion_noise(self, tokens, p, source_idx=None):
        if p == 0.0:
            return tokens, source_idx

        num_tokens = len(tokens)
        n = int(math.ceil(num_tokens * p))

        noise_indices = torch.randperm(num_tokens + n - 2)[:n] + 1
        try :
            while torch.sum(source_idx[noise_indices]) == 0:
               noise_indices = torch.randperm(num_tokens + n - 2)[:n] + 1
        except:
            #print('nooooooo, add_insert')
            return tokens, source_idx
        noise_mask = torch.zeros(size=(num_tokens + n,), dtype=torch.bool)

        noise_mask[noise_indices] = 1
        result = torch.LongTensor(n + len(tokens)).fill_(-1)
        result_idx=torch.IntTensor(n + len(tokens)).fill_(0)
        #print('result_idx{}, result_idx{}'.format(result, result_idx))

        num_random = int(math.ceil(n * self.random_ratio))
        result[noise_indices[num_random:]] = self.mask_idx
        result[noise_indices[:num_random]] = torch.randint(
            low=1, high=len(self.vocab), size=(num_random,)
        )
        
        result[~noise_mask] = tokens
        result_idx[~noise_mask] = source_idx

        assert (result >= 0).all()
        return result, result_idx

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.
        Args:
            samples (List[dict]): samples to collate
        Returns:
            dict: a mini-batch of data
        """
        return collate(
            samples, self.vocab.pad(), self.eos, self.vocab, pad_to_length=pad_to_length
        )

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return self.sizes[index]

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return self.sizes[index]

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        return indices[np.argsort(self.sizes[indices], kind="mergesort")]

    def prefetch(self, indices):
        self.src.prefetch(indices)
        self.tgt.prefetch(indices)

    @property
    def supports_prefetch(self):
        return (
            hasattr(self.src, "supports_prefetch")
            and self.src.supports_prefetch
            and hasattr(self.tgt, "supports_prefetch")
            and self.tgt.supports_prefetch
        )

