from copy import deepcopy
import numpy as np

class Dataset:

    def __init__(self, train_X, train_y, val_X, val_y):
        self.train_X = train_X
        self.train_y = train_y
        self.val_X = val_X
        self.val_y = val_y



class Trainer:

    def __init__(self, model, dataset, optim,
                 num_epochs=20,
                 batch_size=20,
                 learning_rate=1e-3,
                 learning_decay = 1):
        self.dataset = dataset
        self.model = model
        self.optim = optim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.optimizers = None
        self.learning_decay = learning_decay


    def setup_optimizers(self):
        params = self.model.params()
        self.optimizers = {}
        for param_name, param in params.items():
            self.optimizers[param_name] = deepcopy(self.optim)


    def fit(self):


        if self.optimizers is None:
            self.setup_optimizers()

        num_train = self.dataset.train_X.shape[0]
        best_val_loss = []
        train_loss_history = []
        val_loss_history = []


        for epoch in range(self.num_epochs):

            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(self.batch_size, num_train, self.batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)
            batch_losses = np.zeros(len(batches_indices))

            if (epoch > 250):
                self.learning_rate *= self.learning_decay
            for batch_id, batch_indices in (enumerate(batches_indices)):
                batch_X = self.dataset.train_X[batch_indices]
                batch_y = self.dataset.train_y[batch_indices]



                out = self.model.forward(batch_X)
                train_loss = np.mean((out-batch_y)**2)

                grad = out-batch_y



                for param in self.model.params().values():
                    param.grad.fill(0)
                self.model.backward(grad)

                for param_name, param in self.model.params().items():
                    optimizer = self.optimizers[param_name]
                    optimizer.update(param.value, param.grad, self.learning_rate)

                batch_losses[batch_id] = train_loss



            val_out = self.model.forward(self.dataset.val_X)
            val_loss = np.mean((val_out-self.dataset.val_y)**2)
            val_loss_history.append(val_loss)
            train_loss_history.append(batch_losses.mean())
            if val_loss<best_val_loss:
                best_val_loss = val_loss
            # print(f'New best val loss: {val_loss}')
            for param_name, param in self.model.params().items():
                np.save(f'best_{self.model.name}_model/{param_name}.npy', param.value)



            if epoch % 50 == 0:
                print(f'Epoch {epoch}:  Train loss: {batch_losses.mean():.5f}  Val loss: {val_loss:.5f}')


        return train_loss_history, val_loss_history
