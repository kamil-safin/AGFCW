import torch
import numpy as np
from time import time
from sklearn.metrics import accuracy_score

class AGFFull:

    def __init__(self, model, dataset, batch_selector, n_epoch, L_0, D_0=3*1e-2, max_iter=300, tau=7*1e-7, min_L=1, calc_D=False, accelerated=False) -> None:
        self.model = model
        self.dataset = dataset
        self.batch_selector = batch_selector
        self.n_epoch = n_epoch
        self.calc_D = calc_D
        self.accelerated = accelerated
        self.L = [L_0]
        self.D = [D_0]
        self.min_L = min_L
        self.tau = tau
        self.max_iter = max_iter
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
        if self.accelerated:
            self.A = 0
            u = []
            for i in range(self.model.n_params):
                u.append(float(self.model.get_param(i)))
            self.u = np.array(u)
        while epoch_counter < self.n_epoch:
            L = self.L[-1] / 4
            dec_condition = False
            while not dec_condition:
                L = 2*L
                if self.calc_D:
                    self._estimate_D()
                dec_condition = self.check_condition(L)
                if self.epoch_is_done:
                    epoch_counter += 1
            iter_counter += 1
            if iter_counter > self.max_iter:
                break
            train_loss = self.model.get_loss(self.dataset.X_train, self.dataset.y_train)
            test_loss = self.model.get_loss(self.dataset.X_test, self.dataset.y_test)
            self.train_acc.append(self._calc_accuracy(torch.FloatTensor(self.dataset.X_train), self.dataset.y_train))
            self.test_acc.append(self._calc_accuracy(torch.FloatTensor(self.dataset.X_test), self.dataset.y_test))
            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)
            self.time_hist.append(time() - start_time)
            print(f'\r{iter_counter} iters, {epoch_counter} epoches complete, train loss: {train_loss:.3}, test loss: {test_loss:.3}', end='')
            self.L.append(L)

    def check_condition(self, L):
        if self.accelerated:
            alpha = (1 + np.sqrt(1 + 4*self.A*L)) / 2*L
            new_A = self.A + alpha
            batch_idxs, self.epoch_is_done = self.batch_selector.get_batch_idxs(D=self.D[-1], L=L, alpha=alpha)
            self.batch_sizes.append(len(batch_idxs))
            X_batch, y_batch = self.dataset.get_train_data(batch_idxs)
            mult = self.A / new_A
            add = alpha*self.u / new_A
            for i in range(self.model.n_params):
                upd_param = self.model.get_param(i)
                upd_param *= mult
                upd_param += add[i]
            grad = self._calc_grad(X_batch, y_batch)
            self.u -= alpha*grad
            new_add = alpha*self.u / new_A
            for i in range(self.model.n_params):
                upd_param = self.model.get_param(i)
                upd_param -= add[i]
                upd_param += new_add[i]
            self.A = new_A
        else:
            batch_idxs, self.epoch_is_done = self.batch_selector.get_batch_idxs(D=self.D[-1], L=L)
            self.batch_sizes.append(len(batch_idxs))
            X_batch, y_batch = self.dataset.get_train_data(batch_idxs)
            grad = self._calc_grad(X_batch, y_batch)
            step = grad / (2 * L)
            self._make_step(step)
        loss_after_step = self.model.get_loss(X_batch, y_batch)
        grad_norm = np.sum(grad**2)
        dec_condition = loss_after_step < self.curr_loss - grad_norm**2/(4*L) + self.epsilon/2
        if not dec_condition:
            if self.accelerated:
                for i in range(self.model.n_params):
                    upd_param = self.model.get_param(i)
                    upd_param -= new_add[i]
                    upd_param /= mult
            else:
                self._make_step(-step)
        return dec_condition

    def _calc_grad(self, X, y):
        self.curr_loss = self.model.get_loss(X, y)
        grad = np.zeros(self.model.n_params)
        for param_idx in range(self.model.n_params):
            param = self.model.get_param(param_idx)
            param += self.tau
            loss_after = self.model.get_loss(X, y)
            param -= self.tau
            grad[param_idx] = ((loss_after - self.curr_loss) / self.tau)
        return grad

    def _make_step(self, step):
        for param_idx in range(self.model.n_params):
            param = self.model.get_param(param_idx)
            param -= step[param_idx]

    def _estimate_D(self):
        if len(self.batch_sizes) == 0:
            return
        batch_idxs, _ = self.batch_selector.get_batch_idxs(shift_pointer=False, batch_size=self.batch_sizes[-1]//2)
        X_batch, y_batch = self.dataset.get_train_data(batch_idxs)
        gradients = []
        for item, label in zip(X_batch, y_batch):
            gradients.append(self._calc_grad(item.reshape(1, -1), label.reshape(1, -1)))
        self.D.append(np.var(gradients, axis=0).sum())

    def _calc_accuracy(self, X, y_true):
        scores = self.model(X)
        y_pred = np.argmax(scores.detach().numpy(), axis=1)
        return accuracy_score(y_true, y_pred)