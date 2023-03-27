# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II


@dataclass
class CrossEntropyCriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")


@register_criterion("cross_entropy", dataclass=CrossEntropyCriterionConfig)
class CrossEntropyCriterion(FairseqCriterion):
    def __init__(self, task, sentence_avg):
        super().__init__(task)
        self.sentence_avg = sentence_avg

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )

        if sample.get("net_input_mask", None) is not None:
            net_output2 = model(**sample["net_input_mask"])
            loss_all, loss_dec, loss_dec2, loss_enc_wso,  loss_enc_kl = self.compute_loss(model, net_output, net_output2, sample, reduce=reduce)
            logging_output = {
                "loss": loss_all.data,
                "encoder-loss-wso": loss_enc_wso.data,
                "encoder-loss-kl": loss_enc_kl.data,
                "decoder-loss": loss_dec.data,
                "decoder-loss-mask": loss_dec2.data,
                "ntokens": sample["ntokens"],
                "nsentences": sample["target"].size(0),
                "sample_size": sample_size,
            }
        elif sample.get("wso_labels", None) is not None:
            loss_all, loss_dec, loss_enc_wso = self.compute_loss(model, net_output, sample = sample, reduce=reduce)
            logging_output = {
                "loss": loss_all.data,
                "encoder-loss-wso": loss_enc_wso.data,
                "decoder-loss": loss_dec.data,
                "ntokens": sample["ntokens"],
                "nsentences": sample["target"].size(0),
                "sample_size": sample_size,
            }
        else:
            loss_all = self.compute_loss(model, net_output, sample = sample, reduce=reduce)
            logging_output = {
                "loss": loss_all.data,
                "ntokens": sample["ntokens"],
                "nsentences": sample["target"].size(0),
                "sample_size": sample_size,
            }

        return loss_all, sample_size, logging_output

    def compute_loss(self, model, net_output, net_output2 = None, sample =None, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)

        loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction="sum" if reduce else "none",
        )

        if sample.get("wso_labels", None) is not None:
            wso_labels=sample['wso_labels'].view(-1)
            encoder_lprobs = net_output[-2]
            encoder_lprobs = utils.log_softmax(encoder_lprobs, dim=-1)
            encoder_lprobs= encoder_lprobs.view(-1, encoder_lprobs.size(-1))
            loss_wso=F.nll_loss(encoder_lprobs,wso_labels.long(),ignore_index=self.padding_idx, reduction="sum" if reduce else "none",)
        
            if net_output2 != None:
                lprobs2 = model.get_normalized_probs(net_output2, log_probs=True)
                lprobs2 = lprobs2.view(-1, lprobs2.size(-1))
                loss2 = F.nll_loss(
                    lprobs2,
                    target,
                    ignore_index=self.padding_idx,
                    reduction="sum" if reduce else "none",
                )
                enc_mask_out= F.avg_pool1d(net_output[-1].transpose(0,1).transpose(1,2), net_output[-1].shape[0]).squeeze(2)
                enc_mask_out2= F.avg_pool1d(net_output2[-1].transpose(0,1).transpose(1,2), net_output2[-1].shape[0]).squeeze(2)
                sample_size = (sample["target"].size(0) if self.sentence_avg else sample["ntokens"])
                loss_kl = self._contrastive_loss_forward(enc_mask_out, enc_mask_out2)*int(sample_size/enc_mask_out.size(0))
                loss_all = 0.5*loss+0.5*loss2+0.05*loss_kl+0.05*loss_wso
                return loss_all, loss, loss2, loss_wso, loss_kl
            else:
                loss_all = loss+0.05*loss_wso
                return loss_all, loss, loss_wso
        else:
            return loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        #print('here here',logging_outputs)
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)

        if logging_outputs[0].get("encoder-loss-wso", None) is not None:
            enc_loss_sum_wso = sum(log.get("encoder-loss-wso", 0) for log in logging_outputs)
            dec_loss_sum = sum(log.get("decoder-loss", 0) for log in logging_outputs)
            if logging_outputs[0].get("encoder-loss-kl", None) is not None:
                enc_loss_sum_kl = sum(log.get("encoder-loss-kl", 0) for log in logging_outputs)
                dec_loss_sum_mask = sum(log.get("decoder-loss-mask", 0) for log in logging_outputs)

        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if logging_outputs[0].get("encoder-loss-wso", None) is not None:
            metrics.log_scalar("enc_loss_wso", enc_loss_sum_wso / sample_size / math.log(2), sample_size, round=3)
            metrics.log_scalar("dec_loss", dec_loss_sum / sample_size / math.log(2), sample_size, round=3)
            if logging_outputs[0].get("encoder-loss-kl", None) is not None:
                metrics.log_scalar("enc_loss_kl", enc_loss_sum_kl / sample_size / math.log(2), sample_size, round=3)
                metrics.log_scalar("dec_loss-mask", dec_loss_sum_mask / sample_size / math.log(2), sample_size, round=3)

        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True

    def _contrastive_loss_forward(self,
                                  hidden1: torch.Tensor,
                                  hidden2: torch.Tensor,
                                  hidden_norm: bool = True,
                                  temperature: float = 1.0):
        """
        hidden1/hidden2: (bsz, dim)
        """
        batch_size, hidden_dim = hidden1.shape
        LARGE_NUM = 1e9
        if hidden_norm:
            hidden1 = torch.nn.functional.normalize(hidden1, p=2, dim=-1)
            hidden2 = torch.nn.functional.normalize(hidden2, p=2, dim=-1)

        hidden1_large = hidden1
        hidden2_large = hidden2
        labels = torch.arange(0, batch_size).to(device=hidden1.device)
        masks = torch.nn.functional.one_hot(torch.arange(0, batch_size), num_classes=batch_size).to(device=hidden1.device, dtype=torch.float)

        logits_aa = torch.matmul(hidden1, hidden1_large.transpose(0, 1)) / temperature  # shape (bsz, bsz)
        logits_aa = logits_aa - masks * LARGE_NUM
        logits_bb = torch.matmul(hidden2, hidden2_large.transpose(0, 1)) / temperature  # shape (bsz, bsz)
        logits_bb = logits_bb - masks * LARGE_NUM
        logits_ab = torch.matmul(hidden1, hidden2_large.transpose(0, 1)) / temperature  # shape (bsz, bsz)
        logits_ba = torch.matmul(hidden2, hidden1_large.transpose(0, 1)) / temperature  # shape (bsz, bsz)

        loss_a = torch.nn.functional.cross_entropy(torch.cat([logits_ab, logits_aa], dim=1), labels,reduction="sum")
        loss_b = torch.nn.functional.cross_entropy(torch.cat([logits_ba, logits_bb], dim=1), labels,reduction="sum")
        loss = (loss_a + loss_b)/2
        return loss
