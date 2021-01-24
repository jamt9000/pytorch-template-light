import torch.nn.functional as F

def binary_cross_entropy(input, meta):
    target = meta['target'].reshape(input.shape)
    return F.binary_cross_entropy_with_logits(input, target)

def cross_entropy(input, meta):
    return F.cross_entropy(input, meta['target'])
