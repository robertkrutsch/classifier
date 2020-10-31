import torch
from network import Network
import torch.optim as optim
from data_loader import Dataset
from network import Network


class NetworkTool(object):

    def __init__(self, path):
        """

        :param path: This is the path where to save the file. Files saved are model, params and checkpoint_xx.pt
        """
        self.path_model = path + 'model.pt'
        self.path_params = path + 'params.pt'
        self.path_checkpoint = path + 'checkpoint_'

    def load_model(self):
        """
        Loading the model with parameters.
        :return: return the model loaded from the model file present at specified path.
        """
        model = torch.load(self.path_model)
        return model

    def load_params(self):
        """
        Load the model parameters into an existing model. This is more flexible than loading model.
        :return: return the model loaded from the model file present at specified path.
        """
        model = Network()
        model.load_state_dict(torch.load(self.path_params))
        return model

    def save_params(self, model):
        """
        Save the model parameters.
        :param model: save the model params into params.pt
        """
        torch.save(model.state_dict(), self.path_params)

    def save_model(self, model):
        """
        Save the model into model.pt. This is not so flexible as loading just parameters.
        :param model: Save the model into model.pt
        """
        torch.save(model, self.path_model)

    def save_checkpoint(self, model, epoch, loss, optimizer):
        """
        Save a checkpoint.
        :param model: model to be saved
        :param epoch: epoch where we are with training
        :param loss: the loss where we are
        :param optimizer: optimizer setup
        """
        self.path_checkpoint = self.path_checkpoint + str(epoch) + '.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss}, self.path_checkpoint)

    def load_checkpoint(self, epoch):
        """
        Load a checkpoint.
        :param epoch: epoch checkpoint that we should load
        :return: model, optimizer , epoch and loss are returned
        """
        self.path_checkpoint = self.path_checkpoint + str(epoch) + '.pt'
        model = Network()
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01,
                               amsgrad=False)

        checkpoint = torch.load(self.path_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        return model, optimizer, epoch, loss

    def run_checkpoint(self, epoch):
        """
        Load a checkpoint file and run the model on the test dataset.
        :param epoch: epoch checkpoint that we should run.
        """
        model, optimizer, epoch, loss = self.load_checkpoint(epoch)
        model.eval()
        test_dataset = Dataset(csv_file='/content/dataset/test.csv',
                               root_dir='/content/dataset/test')

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)

        total = 0
        correct = 0
        for i, data in enumerate(test_loader, 0):
            local_labels = data['labels'].data.numpy()
            local_batch = data['image'].float()
            local_batch = local_batch.to(device)
            model.to(device)
            outputs = model(local_batch)
            _, predicted = torch.max(outputs.data, 1)
            pred_cpu = predicted.cpu().data.numpy()  # move data to CPU
            torch.cuda.synchronize()  # sync the gpu to get the data quicker
            total += 4
            correct += (pred_cpu == local_labels).sum().item()
            print(pred_cpu, local_labels, correct)

        print('Accuracy of the network : %d %%' % (100 * correct / total))
