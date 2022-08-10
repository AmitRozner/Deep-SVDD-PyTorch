from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from base.base_net import BaseNet
from sklearn.metrics import roc_auc_score
from torch.autograd import Variable
import logging
import time
import torch
import torch.optim as optim
import numpy as np


class AETrainer(BaseTrainer):

    def __init__(self, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150, lr_milestones: tuple = (),
                 batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda', n_jobs_dataloader: int = 0,
                 lambda_val: float = 0.1, use_stochastic_gates=False, sigma_gates=1):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)
        self.lambda_val = lambda_val
        self.use_stochastic_gates = use_stochastic_gates
        self.sigma_gates = sigma_gates
        self.train_gates = None

    def train(self, dataset: BaseADDataset, ae_net: BaseNet):
        # logger = logging.getLogger()

        # Set device for network
        ae_net = ae_net.to(self.device)

        # Get train data loader
        train_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        if self.use_stochastic_gates:
            self.train_gates = Gates(train_loader.batch_sampler.sampler.data_source.indices, self.sigma_gates)
            gates_optimizer = optim.Adam(self.train_gates.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                                         amsgrad=self.optimizer_name == 'amsgrad')

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(ae_net.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                               amsgrad=self.optimizer_name == 'amsgrad')


        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Training
        # logger.info('Starting pretraining...')
        # start_time = time.time()
        ae_net.train()
        for epoch in range(self.n_epochs):

            scheduler.step()
            # if epoch in self.lr_milestones:
            #     logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            loss_epoch = 0.0
            n_batches = 0
            # epoch_start_time = time.time()
            for data in train_loader:
                inputs, _, sample_idx = data
                if self.use_stochastic_gates:
                    curr_gates = self.train_gates.forward(sample_idx)
                else:
                    curr_gates = torch.tensor(1).cuda()
                    self.lambda_val = 0

                inputs = inputs.to(self.device)

                # Zero the network parameter gradients
                optimizer.zero_grad()

                if self.use_stochastic_gates:
                    gates_optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                outputs = ae_net(inputs)
                scores = torch.sum((outputs - inputs) ** 2, dim=tuple(range(1, outputs.dim())))
                scores_times_gates = scores * curr_gates
                loss = torch.mean(scores_times_gates) - self.lambda_val * torch.sum(curr_gates)
                loss.backward()
                optimizer.step()

                if self.use_stochastic_gates:
                    gates_optimizer.step()

                loss_epoch += loss.item()
                n_batches += 1

            # log epoch statistics
            # epoch_train_time = time.time() - epoch_start_time
            # logger.info('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'
            #             .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches))

        # pretrain_time = time.time() - start_time
        # logger.info('Pretraining time: %.3f' % pretrain_time)
        # logger.info('Finished pretraining.')

        return ae_net

    def test(self, dataset: BaseADDataset, ae_net: BaseNet):
        logger = logging.getLogger()

        # Set device for network
        ae_net = ae_net.to(self.device)

        # Get test data loader
        _, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Testing
        # logger.info('Testing autoencoder...')
        loss_epoch = 0.0
        n_batches = 0
        # start_time = time.time()
        idx_label_score = []
        ae_net.eval()
        with torch.no_grad():
            for data in test_loader:
                inputs, labels, idx = data
                inputs = inputs.to(self.device)
                outputs = ae_net(inputs)
                scores = torch.sum((outputs - inputs) ** 2, dim=tuple(range(1, outputs.dim())))
                loss = torch.mean(scores)

                # Save triple of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))

                loss_epoch += loss.item()
                n_batches += 1

        # logger.info('Test set Loss: {:.8f}'.format(loss_epoch / n_batches))

        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)

        auc = roc_auc_score(labels, scores)
        # logger.info('Test set AUC: {:.2f}%'.format(100. * auc))

        # test_time = time.time() - start_time
        # logger.info('Autoencoder testing time: %.3f' % test_time)
        # logger.info('Finished testing autoencoder.')

class Gates(torch.nn.Module):
    def __init__(self, indices, sigma_gates):
        super(Gates, self).__init__()
        self.num_of_gates = len(indices)
        self.mu = torch.nn.Parameter(0.5 * torch.ones([self.num_of_gates], requires_grad=True).cuda()) #Variable
        self.sigma_gates = sigma_gates
        self.indices = np.array(indices)

    def forward(self, curr_samples_ind):
        chosen_gates = self.get_curr_mu(curr_samples_ind)
        unbounded_gates = chosen_gates + torch.normal(mean=torch.zeros_like(chosen_gates), std=self.sigma_gates).cuda()
        clamped_gates = torch.clamp(unbounded_gates, min=0, max=1)
        return clamped_gates

    def get_curr_mu(self, curr_samples_ind):
        relevant_indices = []
        for i in curr_samples_ind:
            relevant_indices.append(np.argwhere(self.indices == i.cpu().detach().numpy())[0][0])
        chosen_gates = self.mu[relevant_indices]
        return chosen_gates