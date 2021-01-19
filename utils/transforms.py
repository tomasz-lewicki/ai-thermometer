def img2euc(x, y):
    """
    image frame to Euclidean frame
    """
    x = x - 0.5
    y = -y + 0.5
    return x, y


def euc2img(x, y):
    """
    Euclidean frame to image frame
    """
    x = x + 0.5
    y = -y + 0.5
    return x, y


def shift(x, y, dx, dy):
    x += dx
    y += dy
    return x, y