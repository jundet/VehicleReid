
class FocalLoss(nn.Module):
    """
    Focal loss: focus more on hard samples
    """

    def __init__(self,
                 gamma=0,
                 eps=1e-7):
        """
        :param gamma:
        :param eps:
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        """
        :param input:
        :param target:
        :return:
        """
        log_p = self.ce(input, target)
        p = torch.exp(-log_p)
        loss = (1.0 - p) ** self.gamma * log_p
        return loss.mean()