from pdb import set_trace as bp
import torch.nn.functional as F
import torch


def binary_loss(output, target):

    global_mask, attr_mask, match = target

    global_loss = bce_loss(output[global_mask], match[global_mask])
    attr_loss = bce_loss(output[attr_mask], match[attr_mask])
    return 0.5 * global_loss + 0.5 * attr_loss


def global_binary_loss(output, target):
    global_mask, attr_mask, match = target
    global_loss = bce_loss(output[global_mask], match[global_mask])
    return global_loss


def attr_binary_loss(output, target):
    global_mask, attr_mask, match = target
    attr_loss = bce_loss(output[attr_mask], match[attr_mask])
    return attr_loss


def sim_loss(embs_logits):
    target = torch.arange(len(embs_logits)).to(embs_logits.get_device())
    loss1 = F.cross_entropy(embs_logits, target)
    loss2 = F.cross_entropy(embs_logits.t(), target)
    return (loss1 + loss2) / 2


def attr_loss(attrs_pred, attrs_mask, target):
    losses = torch.cat(
        [
            F.cross_entropy(
                attrs_pred[i][attrs_mask[:, i]],
                target[:, i][attrs_mask[:, i]],
                reduction='none'
            )
            for i in range(12)
        ]
    )
    loss = torch.mean(losses)
    return loss


def sim_attr_loss(output, target):
    embs_logit, attrs_pred = output
    attrs_match_mask, _, attrs_label = target
    return sim_loss(embs_logit) + attr_loss(attrs_pred, attrs_match_mask, attrs_label)


def bce_loss(output, target):
    return F.binary_cross_entropy(output, target)


def bce_loss_with_logits(output, target):
    return F.binary_cross_entropy_with_logits(output, target)


def cc_loss_with_logits(output, target):
    return F.cross_entropy(output, target)


def soft_cross_entropy_loss(output, target):
    """
    src: https://blog.csdn.net/Hungryof/article/details/93738717
    """
    return torch.sum(torch.mul(-F.log_softmax(output, dim=1), target)) / output.shape[0]


def seq2seq_drop_zero(output, target):
    output = output[:, :-1]
    # set_trace()
    output = output.contiguous().view(-1, int(output.size()[-1]))
    target = target.contiguous().view(-1, int(target.size()[-1]))
    # drop all zero target
    indices = (target.sum(dim=1) != 0).type(torch.bool)
    target = target[indices]
    output = output[indices]
    return output, target
