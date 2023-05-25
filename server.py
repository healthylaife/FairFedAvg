from typing import List, Tuple
import utils
import flwr as fl
import json
from flwr.common import Metrics
from FairFedAvg import FairFedAvg
import statistics
import torch

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = []
    examples = []
    num_sen = 5
    tp = [0] * num_sen
    fp = [0] * num_sen
    tn = [0] * num_sen
    fn = [0] * num_sen
    for num_examples, metrics in metrics:
        accuracies.append(metrics['accuracy'] * num_examples)
        examples.append(num_examples)
        metrics['acc_sens'] = json.loads(metrics['acc_sens'].decode('utf-8'))
        print(metrics['acc_sens'])
        for i in metrics['acc_sens']:
            tp[int(i)] += metrics['acc_sens'][i]['tp']
            fp[int(i)] += metrics['acc_sens'][i]['fp']
            tn[int(i)] += metrics['acc_sens'][i]['tn']
            fn[int(i)] += metrics['acc_sens'][i]['fn']


    tpr = []
    fpr = []
    acc = []
    for i in range(num_sen):
        if tp[i] + fn[i] > 0:
            tpr.append(tp[i] / (tp[i] + fn[i]))
        if fp[i] + tn[i] > 0:
            fpr.append(fp[i] / (fp[i] + tn[i]))
        acc.append((tp[i] + tn[i]) / (tp[i] + fn[i] + fp[i] + tn[i]))
    return_dict = {"accuracy": sum(accuracies) / sum(examples),
                   'eosd' : statistics.stdev(tpr),
                   'wtpr' : min(tpr),
                   'apsd' : statistics.stdev(acc)}
    # Aggregate and return custom metric (weighted average)
    return return_dict

model = utils.Dipole(60)

model_parameters = [val.cpu().numpy() for _, val in model.state_dict().items()]
# Define strategy
strategy = FairFedAvg(
                evaluate_metrics_aggregation_fn=weighted_average,
                initial_parameters = fl.common.ndarrays_to_parameters(model_parameters),
                min_fit_clients=5,
                min_evaluate_clients=5,
                min_available_clients=5
                )



# Start Flower server
fl.server.start_server(
    server_address="127.0.0.1:8080",
    config=fl.server.ServerConfig(num_rounds=8),
    strategy=strategy
)
