import copy

from server import Server
from client import *
import datasets

if __name__ == '__main__':

    with open("../utils/conf.json", 'r') as f:
        conf = json.load(f)

    train_datasets, eval_datasets = datasets.get_dataset("../data/", conf["type"])

    server = Server(conf, eval_datasets)

    clients = []

    # non-IID数据
    client_idx = datasets.dirichlet_nonIID_data(train_datasets, conf)

    # print(client_idx)

    for c in range(conf["clients"]):
        clients.append(Client(conf, server.global_model, train_datasets, client_idx[c + 1], c + 1))

    server.set_clients(clients)

    all_acc = []
    for e in range(conf["global_epochs"]):
        print("Global Epoch %d" % e)
        candidates = clients

        server.n_data = 0
        client_state = {}
        client_n_data = {}
        client_coeff = {}
        client_norm_grad = {}

        for c in candidates:
            state_dict, n_data, coeff, norm_grad = c.local_train(server.global_model)
            client_state[c.client_id] = state_dict
            client_n_data[c.client_id] = n_data
            client_coeff[c.client_id] = coeff
            client_norm_grad[c.client_id] = norm_grad
            server.n_data += n_data

        server.model_aggregate(client_state, client_n_data, client_coeff, client_norm_grad)

        acc, loss = server.model_eval()
        all_acc.append(acc)
        print("Global Epoch %d, acc: %f, loss: %f\n" % (e, acc, loss))

    print(all_acc)
