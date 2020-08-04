import torch


class Metric:
    '''Metric computes accuracy/precision/recall/confusion_matrix with batch updates.'''

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.y = []
        self.t = []

    def update(self, y, t):
        '''Update with batch outputs and labels.

        Args:
          y: (tensor) model outputs sized [N,].
          t: (tensor) labels targets sized [N,].
        '''
        self.y.append(y)
        self.t.append(t)

    def _process(self, y, t):
        '''Compute TP, FP, FN, TN.

        Args:
          y: (tensor) model outputs sized [N,].
          t: (tensor) labels targets sized [N,].

        Returns:
          (tensor): TP, FP, FN, TN, sized [num_classes,].
        '''
        tp = torch.empty(self.num_classes)
        fp = torch.empty(self.num_classes)
        fn = torch.empty(self.num_classes)
        tn = torch.empty(self.num_classes)
        for i in range(self.num_classes):
            tp[i] = ((y == i) & (t == i)).sum().item()
            fp[i] = ((y == i) & (t != i)).sum().item()
            fn[i] = ((y != i) & (t == i)).sum().item()
            tn[i] = ((y != i) & (t != i)).sum().item()
        return tp, fp, fn, tn

    def accuracy(self, reduction='mean'):
        '''Accuracy = (TP+TN) / (P+N).

        Args:
          reduction: (str) mean or none.

        Returns:
          (tensor) accuracy.
        '''
        if not self.y or not self.t:
            return
        assert(reduction in ['none', 'mean'])
        y = torch.cat(self.y, 0)
        t = torch.cat(self.t, 0)
        tp, fp, fn, tn = self._process(y, t)
        if reduction == 'none':
            acc = tp / (tp + fn)
        else:
            acc = tp.sum() / (tp + fn).sum()
        return acc

    def precision(self, reduction='mean'):
        '''Precision = TP / (TP+FP).

        Args:
          reduction: (str) mean or none.

        Returns:
          (tensor) precision.
        '''
        if not self.y or not self.t:
            return
        assert(reduction in ['none', 'mean'])
        y = torch.cat(self.y, 0)
        t = torch.cat(self.t, 0)
        tp, fp, fn, tn = self._process(y, t)
        prec = tp / (tp + fp)
        prec[torch.isnan(prec)] = 0
        if reduction == 'mean':
            prec = prec.mean()
        return prec

    def recall(self, reduction='mean'):
        '''Recall = TP / P.

        Args:
          reduction: (str) mean or none.

        Returns:
          (tensor) recall.
        '''
        if not self.y or not self.t:
            return
        assert(reduction in ['none', 'mean'])
        y = torch.cat(self.y, 0)
        t = torch.cat(self.t, 0)
        tp, fp, fn, tn = self._process(y, t)
        recall = tp / (tp + fn)
        recall[torch.isnan(recall)] = 0
        if reduction == 'mean':
            recall = recall.mean()
        return recall

    def confusion_matrix(self):
        y = torch.cat(self.y, 0)
        t = torch.cat(self.t, 0)
        matrix = torch.zeros(self.num_classes, self.num_classes)
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                matrix[j][i] = ((y == i) & (t == j)).sum().item()
        return matrix


def test():
    import pytorch_lightning.metrics.functional as M
    nc = 10
    m = Metric(num_classes=nc)
    y = torch.randint(0, nc, (10,))
    t = torch.randint(0, nc, (10,))
    m.update(y, t)

    print('\naccuracy:')
    print(m.accuracy('none'))
    print(M.accuracy(y, t, nc, 'none'))
    print(m.accuracy('mean'))
    print(M.accuracy(y, t, nc, 'elementwise_mean'))

    print('\nprecision:')
    print(m.precision('none'))
    print(M.precision(y, t, nc, 'none'))
    print(m.precision('mean'))
    print(M.precision(y, t, nc, 'elementwise_mean'))

    print('\nrecall:')
    print(m.recall('none'))
    print(M.recall(y, t, nc, 'none'))
    print(m.recall('mean'))
    print(M.recall(y, t, nc, 'elementwise_mean'))

    print('\nconfusion matrix:')
    print(m.confusion_matrix())
    print(M.confusion_matrix(y, t))


if __name__ == '__main__':
    test()
