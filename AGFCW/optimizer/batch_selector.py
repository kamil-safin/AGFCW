

class BatchSelector:

    def __init__(self, n_params, idxs, epsilon=1e-4, coord_wise=True, div=1) -> None:
        self.coord_wise = coord_wise
        self.n_params = n_params
        self.idxs = idxs
        self.n_samples = len(idxs)
        self.epsilon = epsilon
        self._start_idx = 0
        self.div = div

    def _get_batch_size(self, L, D, alpha=None):
        if self.coord_wise:
            if alpha is None:
                batch_size = min(max(int(D*self.n_params / (L*self.epsilon*self.div)), 1), self.n_samples)
            else:
                # accelerated
                batch_size = min(max(int(D*alpha / (self.epsilon*self.div)), 1), self.n_samples)
        else:
            if alpha is None:
                batch_size = min(max(int(D / (L*self.epsilon*self.div)), 1), self.n_samples)
            else:
                # accelerated
                batch_size = min(max(int(D*alpha / self.epsilon*self.div), 1), self.n_samples)
        return batch_size

    def get_batch_idxs(self, D=None, L=None, shift_pointer=True, batch_size=None, alpha=None):
        if batch_size is None:
            batch_size = self._get_batch_size(L, D, alpha)
        end_idx = self._start_idx + batch_size
        if end_idx > self.n_samples:
            end_idx = end_idx % self.n_samples
            batch_idxs = self.idxs[self._start_idx:] + self.idxs[:end_idx]
            epoch_is_done = True
        else:
            batch_idxs = self.idxs[self._start_idx: end_idx]
            epoch_is_done = False
        if shift_pointer:
            self._start_idx = end_idx
        return batch_idxs, epoch_is_done
