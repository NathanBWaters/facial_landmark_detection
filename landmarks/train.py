'''
Models outputs the characters in the game
'''
import os
import wandb
import torch
import torch.optim as optim
from torch.nn import MSELoss
from torchsummary import summary

from landmarks.loader import get_datasets
from landmarks.model import LandmarkModel
from landmarks.constants import SEED, INPUT_DIMS, MODEL_DIR


class TrainCharModel(object):
    '''
    Class for training a model
    '''

    def __init__(self, name, use_wandb=True, print_summary=True):
        '''
        Create the training object
        '''
        self.name = name
        torch.manual_seed(SEED)

        train, val, test = get_datasets()

        self.batch_size = 64

        self.device = 'cpu'
        loader_data = ({'num_workers': 1, 'pin_memory': True}
                       if self.device == 'cuda' else {})
        self.train_loader = torch.utils.data.DataLoader(
            train,
            batch_size=self.batch_size,
            shuffle=True,
            **loader_data
        )
        self.validation_loader = torch.utils.data.DataLoader(
            train,
            batch_size=self.batch_size,
            shuffle=True,
            **loader_data
        )
        self.test_loader = torch.utils.data.DataLoader(
            test,
            batch_size=self.batch_size,
            shuffle=True,
            **loader_data
        )

        self.model = LandmarkModel(load_weights=False)
        self.model.to(self.device)

        self.learning_rate = 3e-5
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate)

        self.epochs = 1000

        self.criterion = MSELoss()

        if print_summary:
            summary(self.model,
                    input_size=(INPUT_DIMS[2], INPUT_DIMS[0], INPUT_DIMS[1]),
                    device=self.device)

        self.use_wandb = use_wandb
        if self.use_wandb:
            print('Using wandb!')
            wandb.init(name=self.name, project='facial_landmarks')
            wandb.watch(self.model)

    def train(self):
        '''
        Training the character model
        '''
        self.model.train()
        correct = 0
        training_loss = 0

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()

            output = self.model(data)

            loss = self.criterion(output, target)

            # computes gradient for all of the parameters w.r.t the loss
            loss.backward()
            # updates the parameters based on the gradient
            self.optimizer.step()

            training_loss += loss.item()

            # say the prediction is accurate if it gets the integer
            # representation of the label correct
            accuracy = (
                (torch.round(output) == torch.round(target)).sum() /
                float(output.numel()))
            correct += accuracy

        num_batches = len(self.train_loader.dataset) / self.batch_size
        training_loss /= num_batches
        score = 100. * correct / num_batches

        stats = {'training_loss': training_loss, 'training_accuracy': score}
        if self.use_wandb:
            print(stats)
            wandb.log(stats)
        else:
            print(stats)

    def evaluate(self, loader, name):
        '''
        Test / Validate the model
        '''
        self.model.eval()
        loss = 0
        correct = 0

        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss += self.criterion(output, target).item()

                accuracy = (
                    (torch.round(output) == torch.round(target)).sum() /
                    float(output.numel()))
                correct += accuracy

        num_batches = len(loader.dataset) / self.batch_size
        loss /= num_batches
        score = 100. * correct / num_batches

        stats = {'{}_loss'.format(name): loss,
                 '{}_accuracy'.format(name): score}
        if self.use_wandb:
            print(stats)
            wandb.log(stats)
        else:
            print(stats)

    def loop(self):
        '''
        This is the main training and testing loop
        '''
        for epoch in range(1, self.epochs + 1):
            print('Epoch is: {}'.format(epoch))
            # train on training set
            self.train()
            # compare against validation set
            self.evaluate(self.validation_loader, 'validation')

            if epoch % 50 == 0:
                # compare against test set
                self.evaluate(self.test_loader, 'test')
                print('Saving model at epoch {}'.format(epoch))
                torch.save(
                    self.model.state_dict(),
                    os.path.join(MODEL_DIR, self.name + '__' + str(epoch)))


if __name__ == '__main__':
    training_class = TrainCharModel(
        '1_simple_lenet',
        use_wandb=True,
    )
    training_class.loop()
    print('done')
