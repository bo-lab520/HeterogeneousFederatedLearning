import json

from SCAFFOLD.server import Server
from client import *
import datasets

if __name__ == '__main__':

    with open("../utils/conf.json", 'r') as f:
        conf = json.load(f)
    train_datasets, eval_datasets = datasets.get_dataset("../data/", conf["type"])

    server = Server(conf, eval_datasets)
    clients = []
    # non-IID数据
    client_idx = datasets.get_nonIID_data(conf)

    # print(client_idx)

    for c in range(conf["clients"]):
        clients.append(Client(conf, server.global_model, train_datasets, client_idx[c + 1], c + 1))

    all_acc = []
    # 记录控制变量差值
    delta_c = copy.deepcopy(server.global_model.state_dict())
    # 记录控制变量差值
    delta_x = copy.deepcopy(server.global_model.state_dict())

    for e in range(conf["global_epochs"]):
        print("Global Epoch %d" % e)

        for ci in delta_c:
            delta_c[ci] = 0.0
        for ci in delta_c:
            delta_x[ci] = 0.0

        candidates = clients
        for c in candidates:
            weights, local_delta_c, local_delta = c.local_train(server.global_model, server.control_global)

            for w in delta_c:
                delta_x[w] += local_delta[w]
                delta_c[w] += local_delta_c[w]

        server.model_aggregate(delta_x)
        server.control_aggregate(delta_c)

        acc, loss = server.model_eval()
        all_acc.append(acc)
        print("Global Epoch %d, acc: %f, loss: %f\n" % (e, acc, loss))

    print(all_acc)
