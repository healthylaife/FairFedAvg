from collections import OrderedDict
import sys
import flwr as fl
import utils
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
import json
import statistics
from sklearn.model_selection import KFold

def train(net, trainloader, epochs):
    """Train the model on the training set."""
    net.to(DEVICE)
    net.train()
    criterion = torch.nn.BCEWithLogitsLoss()
    criterion_sens = torch.nn.CrossEntropyLoss()


    enc_vars = [i for i in net.embed_func.parameters()] + [i for i in net.rnn_model.parameters()] + [i for i in net.attention_func.parameters()] + [i for i in net.output_activate.parameters()] + [i for i in net.output_func.parameters()]
    enc_class_vars = enc_vars + [v for v in net.classifier.parameters()]
    sens_vars = [v for v in net.sens_class.parameters()]

    optimizer = torch.optim.AdamW(enc_class_vars, lr=0.001)
    optimizer_sens = torch.optim.AdamW(sens_vars, lr=0.002)

    for _ in range(epochs):
        correct, total = 0, 0
        confusion = {}
        loss_c, loss_s = 0, 0
        for i in range(5):
            confusion[i] = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
        step = 0
        for attr, sens, labels, m in tqdm(trainloader):
            optimizer.zero_grad()
            optimizer_sens.zero_grad()
            sens = sens.to(DEVICE).squeeze(dim=1)
            labels = labels.to(DEVICE).squeeze(dim=1)
            rep, outputs, sens_clas = net(attr.to(DEVICE), m)
            outputs, sens_clas = outputs.squeeze(dim=1), sens_clas.squeeze(dim=1)

            alpha = 0.5
            loss_class = (1 - alpha) * criterion(outputs, labels) - alpha * criterion_sens(sens_clas, sens)  # + f_m
            loss_sens = criterion_sens(sens_clas, sens)

            loss_class.backward(retain_graph=True)
            loss_sens.backward()

            optimizer.step()
            optimizer_sens.step()

            outputs = torch.sigmoid(outputs)
            total += labels.size(0)
            correct += ((outputs >= 0.5) == labels).sum().item()

            loss_c += loss_class.item()
            loss_s += loss_sens.item()

            for i in range(5):
                confusion[i]['tp'] += ((sens[:] == i) * (labels == 1) * ((outputs >= 0.5) == labels)).sum().item()
                confusion[i]['fp'] += ((sens[:] == i) * (labels == 0) * ((outputs >= 0.5) != labels)).sum().item()
                confusion[i]['tn'] += ((sens[:] == i) * (labels == 0) * ((outputs >= 0.5) == labels)).sum().item()
                confusion[i]['fn'] += ((sens[:] == i) * (labels == 1) * ((outputs >= 0.5) != labels)).sum().item()
            step += 1
        fair_f = []
        acc_f = []
        for i in confusion:
            if confusion[i]['tp'] + confusion[i]['fn'] > 0:
                fair_f.append(confusion[i]['tp'] / (confusion[i]['tp'] + confusion[i]['fn']))
            acc_f.append((confusion[i]['tp'] + confusion[i]['tn']) / (confusion[i]['tp'] + confusion[i]['fn'] + confusion[i]['fp'] + confusion[i]['tn']))
        eosd = statistics.stdev(fair_f)
        wtpr = min(fair_f)
        apsd = statistics.stdev(acc_f)
        print(loss_c / step, loss_s / step, eosd)
    net.to('cpu')
    return correct / total, eosd, wtpr, apsd


def test(net, testloader):
    """Validate the model on the test set."""
    net.to(DEVICE)
    net.eval()
    criterion = torch.nn.BCEWithLogitsLoss()

    correct, total, loss = 0, 0, 0.0
    acc = {}
    for i in range(5):
        acc[i] = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
    with torch.no_grad():
        for attr, sens, labels, m in tqdm(testloader):
            _, outputs, _ = net(attr.to(DEVICE), m)
            outputs = outputs.squeeze(dim=1)
            sens = sens.to(DEVICE).squeeze(dim=1)
            labels = labels.to(DEVICE).squeeze(dim=1)
            loss += criterion(outputs, labels).item()
            outputs = torch.sigmoid(outputs)
            total += labels.size(0)
            correct += ((outputs >= 0.5) == labels).sum().item()
            for i in range(5):
                acc[i]['tp'] += ((sens[:] == i) * (labels == 1) * ((outputs >= 0.5) == labels)).sum().item()
                acc[i]['fp'] += ((sens[:] == i) * (labels == 0) * ((outputs >= 0.5) != labels)).sum().item()
                acc[i]['tn'] += ((sens[:] == i) * (labels == 0) * ((outputs >= 0.5) == labels)).sum().item()
                acc[i]['fn'] += ((sens[:] == i) * (labels == 1) * ((outputs >= 0.5) != labels)).sum().item()
    net.to('cpu')
    return loss / len(testloader.dataset), correct / total, acc


def load_data(id, k):
    file = 'data/mimic/' + id + '.npz'
    X = np.load(file, mmap_mode='r', allow_pickle=True)
    dset = X['x_train_full']
    fixed = X['fixed_data_train']
    sens = X['sensitive_data']
    dset[dset == np.inf] = 1



    kf = KFold(n_splits=5, random_state=None)
    i = 1
    for train_index, test_index in kf.split(dset):
        train_l = int(len(train_index) * 0.70)
        val_index = train_index[train_l:]
        train_index = train_index[:train_l]

        train_dset = dset[train_index]
        val_dset = dset[val_index]
        test_dset = dset[test_index]

        train_sens = sens[train_index]
        val_sens = sens[val_index]
        test_sens = sens[test_index]

        train_target = fixed[train_index]
        val_target = fixed[val_index]
        test_target = fixed[test_index]

        train_m = fixed[train_index]
        val_m = fixed[val_index]
        test_m = fixed[test_index]

        if i == int(k):
            trainset = utils.DataLoader(train_dset.tolist(), train_sens.tolist(), train_target.tolist(), train_m.tolist())
            valset = utils.DataLoader(val_dset.tolist(), val_sens.tolist(), val_target.tolist(), val_m.tolist())
            testset = utils.DataLoader(test_dset.tolist(), test_sens.tolist(), test_target.tolist(), test_m.tolist())
            break
        i += 1
    trainset = torch.utils.data.DataLoader(dataset=trainset, batch_size=64)
    valset = torch.utils.data.DataLoader(dataset=valset, batch_size=64)
    testset = torch.utils.data.DataLoader(dataset=testset, batch_size=64)

    return trainset, testset, valset, id

# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = utils.Dipole(60, device=DEVICE).to(DEVICE)

trainloader, testloader, valloader, id = load_data(sys.argv[1], sys.argv[2])

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        acc, eosd, wtpr, apsd = train(net, trainloader, epochs=2)
        loss_v, acc_v, sens_acc_v = test(net, valloader)
        return self.get_parameters(config={}), len(trainloader.dataset), {'id': id, 'acc': acc, 'eosd':eosd, 'wtpr':wtpr, 'apsd':apsd, 'acc_val': acc_v, 'sens_acc_val':json.dumps(sens_acc_v).encode('utf-8'), 'num_val': len(valloader.dataset)}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, acc, sens_acc = test(net, testloader)
        return loss, len(testloader.dataset), {"accuracy": acc, 'acc_sens': json.dumps(sens_acc).encode('utf-8')}


# Start Flower client
fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient(),
)
