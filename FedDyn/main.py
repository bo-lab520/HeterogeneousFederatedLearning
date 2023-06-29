import copy
import json

import numpy as np

from server import Server
from client import *
import datasets


def get_mdl_params(model_list, n_par=None):
    if n_par == None:
        exp_mdl = model_list[0]
        n_par = 0
        for name, param in exp_mdl.named_parameters():
            n_par += len(param.data.reshape(-1))

    param_mat = np.zeros((len(model_list), n_par)).astype('float32')
    for i, mdl in enumerate(model_list):
        idx = 0
        for name, param in mdl.named_parameters():
            temp = param.data.cpu().numpy().reshape(-1)
            param_mat[i, idx:idx + len(temp)] = temp
            idx += len(temp)
    return np.copy(param_mat)

if __name__ == '__main__':

    with open("../utils/conf.json", 'r') as f:
        conf = json.load(f)

    train_datasets, eval_datasets = datasets.get_dataset("../data/", conf["type"])

    server = Server(conf, eval_datasets)

    clients = []

    # non-IID数据
    client_idx = datasets.dirichlet_nonIID_data(train_datasets, conf)

    for c in range(conf["clients"]):
        clients.append(Client(conf, server.global_model, train_datasets, client_idx[c], c))

    n_par = len(get_mdl_params([models.get_model("cnnmnist")])[0])
    server.cld_mdl_param = get_mdl_params([server.global_model], n_par)[0]

    local_param_list = np.zeros((conf["clients"], n_par)).astype('float32')
    clnt_params_list = np.zeros((conf["clients"], n_par)).astype('float32')

    all_acc = []
    for e in range(conf["global_epochs"]):
        print("Global Epoch %d" % e)
        candidates = clients

        cld_mdl_param_tensor = torch.tensor(server.cld_mdl_param, dtype=torch.float32)

        weight_accumulator = {}
        for name, params in server.global_model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(params)

        for c in candidates:
            local_param_list_curr = torch.tensor(local_param_list[c.client_id], dtype=torch.float32)

            # diff = c.local_train(server.global_model)
            clnt_models, diff = c.local_train(server.global_model, cld_mdl_param_tensor, local_param_list_curr)
            curr_model_par = get_mdl_params([clnt_models], n_par)[0]

            local_param_list[c.client_id] += curr_model_par - server.cld_mdl_param
            clnt_params_list[c.client_id] = curr_model_par

            for name, params in server.global_model.state_dict().items():
                weight_accumulator[name].add_(diff[name])

        avg_mdl_param = np.mean(clnt_params_list, axis=0)
        server.cld_mdl_param = avg_mdl_param + np.mean(local_param_list, axis=0)

        # server.model_aggregate(avg_mdl_param)
        server.model_aggregate(weight_accumulator)

        acc, loss = server.model_eval()
        all_acc.append(acc)
        print("Global Epoch %d, acc: %f, loss: %f\n" % (e, acc, loss))

    print(all_acc)
