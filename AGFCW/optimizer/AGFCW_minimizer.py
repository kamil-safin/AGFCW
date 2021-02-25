import torch
import numpy as np
from time import time
from sklearn.metrics import accuracy_score

class AGFCW:

    def __init__(self, model, dataset, batch_selector, n_epoch, L_0, D_0=1e-3, max_alpha=800, tau=1e-3, calc_D=False, coord_L=True, accelerated=False) -> None:
        self.model = model
        self.dataset = dataset
        self.batch_selector = batch_selector
        self.n_epoch = n_epoch
        self.calc_D = calc_D
        self.accelerated = accelerated
        self.coord_L = coord_L
        self.max_alpha = max_alpha
        if self.coord_L:
            self.L = [[L_0]]*self.model.n_params
        else:
            self.L = L_0
        self.D = [D_0]
        self.tau = tau
        self.epsilon = self.batch_selector.epsilon
        self.train_losses = []
        self.test_losses = []
        self.batch_sizes = []
        self.train_acc = []
        self.test_acc = []
        self.time_hist = [0]

    def run(self):
        epoch_counter = 0
        iter_counter = 0
        self.train_losses.append(self.model.get_loss(self.dataset.X_train, self.dataset.y_train))
        self.test_losses.append(self.model.get_loss(self.dataset.X_test, self.dataset.y_test))
        start_time = time()
        if not self.coord_L:
            L = self.L
        if self.accelerated:
            self.A = 0
            self.u = []
            for i in range(self.model.n_params):
                self.u.append(float(self.model.get_param(i)))
        while epoch_counter < self.n_epoch:
            dir = np.random.randint(0, self.model.n_params)
            if self.coord_L:
                L = self.L[dir][-1] / 4
            dec_condition = False
            if self.accelerated:
                self.alpha = min((1 + np.sqrt(1 + 4*self.A*L)) / 2*L, self.max_alpha)
                batch_idxs, self.epoch_is_done = self.batch_selector.get_batch_idxs(D=self.D[-1], L=L, alpha=self.alpha)
                X_batch, y_batch = self.dataset.get_train_data(batch_idxs)
            else:
                batch_idxs, self.epoch_is_done = self.batch_selector.get_batch_idxs(D=self.D[-1], L=L)
                X_batch, y_batch = self.dataset.get_train_data(batch_idxs)
            self.batch_sizes.append(len(batch_idxs))
            while not dec_condition:
                L = 2*L
                if self.calc_D:
                    self._estimate_D(dir)
                dec_condition = self.check_condition(dir, L, X_batch, y_batch)
                if self.epoch_is_done:
                    epoch_counter += 1
            iter_counter += 1
            train_loss = self.model.get_loss(self.dataset.X_train, self.dataset.y_train)
            if np.isnan(train_loss):
                break
            test_loss = self.model.get_loss(self.dataset.X_test, self.dataset.y_test)
            self.train_acc.append(self._calc_accuracy(torch.FloatTensor(self.dataset.X_train), self.dataset.y_train))
            self.test_acc.append(self._calc_accuracy(torch.FloatTensor(self.dataset.X_test), self.dataset.y_test))
            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)
            self.time_hist.append(time() - start_time)
            print(f'\r{iter_counter} iters, {epoch_counter} epoches complete, train loss: {train_loss:.3}, test loss: {test_loss:.3}', end='')
            if self.coord_L:
                self.L[dir].append(L)

    def check_condition(self, dir, L, X_batch, y_batch):
        upd_param = self.model.get_param(dir)
        if self.accelerated:
            new_A = self.A + self.alpha
            self.loss_before_step = self.model.get_loss(X_batch, y_batch)
            mult = self.A / new_A
            add = self.alpha*self.u[dir] / new_A
            upd_param *= mult
            upd_param += add
            coord_grad = self._calc_grad(upd_param, X_batch, y_batch)
            upd_param -= add
            self.u[dir] -= self.alpha*coord_grad
            add = self.alpha*self.u[dir] / new_A
            upd_param += add
            self.A = new_A
        else:
            self.loss_before_step = self.model.get_loss(X_batch, y_batch)
            coord_grad = self._calc_grad(upd_param, X_batch, y_batch)
            step = coord_grad / (2 * L)
            upd_param -= step
        self.loss_after_step = self.model.get_loss(X_batch, y_batch)
        dec_condition = self.loss_after_step < self.loss_before_step - coord_grad**2/(4*L) + self.epsilon/2
        if not dec_condition:
            if self.accelerated:
                upd_param -= add
                upd_param /= mult
            else:
                upd_param += step
        return dec_condition

    def _calc_grad(self, upd_param, X, y):
        loss_before = self.model.get_loss(X, y)
        upd_param += self.tau
        loss_after = self.model.get_loss(X, y)
        upd_param -= self.tau
        return (loss_after - loss_before) / self.tau

    def _estimate_D(self, dir):
        if len(self.batch_sizes) == 0:
            return
        batch_idxs, _ = self.batch_selector.get_batch_idxs(shift_pointer=False, batch_size=int(self.batch_sizes[-1]//2))
        X_batch, y_batch = self.dataset.get_train_data(batch_idxs)
        gradients = []
        upd_param = self.model.get_param(dir)
        for i in range(X_batch.shape[0]):
            gradients.append(self._calc_grad(upd_param, X_batch[i].reshape(1, -1), y_batch[i].reshape(1)))
        D = np.var(gradients)
        self.D.append(self.D[-1])

    def _calc_accuracy(self, X, y_true):
        scores = self.model(X)
        y_pred = np.argmax(scores.detach().numpy(), axis=1)
        return accuracy_score(y_true, y_pred)