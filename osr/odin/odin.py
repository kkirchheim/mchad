import torch


def zero_grad(x):
    if type(x) is torch.Tensor():
        torch.fill_(x, 0)


def odin_preprocessing(
    model: torch.nn.Module, criterion, x, y=None, eps=0.05, temperature=1000
):
    """
    TODO: Normalizing the gradient to the same space of image

    Original Odin implementation:
    https://github.com/facebookresearch/odin/blob/master/code/calData.py
    """
    model.apply(zero_grad)

    if y is None:
        with torch.no_grad():
            y = model(x).argmax(dim=1)

    with torch.enable_grad():
        x.requires_grad = True
        logits = model(x) / temperature
        loss = criterion(logits, y)
        loss.backward()
        x_hat = x - eps * x.grad.sign()

    model.apply(zero_grad)

    return x_hat
