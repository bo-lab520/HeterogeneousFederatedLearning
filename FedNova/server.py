import copy

import models, torch


class Server(object):

    def __init__(self, conf, eval_dataset):
        self.conf = conf

        self.n_data = 0

        self.clients = []

        self.global_model = models.get_model(self.conf["model_name"])

        self.eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=self.conf["batch_size"], shuffle=True)

    def set_clients(self, clients):
        self.clients = clients

    def model_aggregate(self, client_state, client_n_data, client_coeff, client_norm_grad):
        model_state = self.global_model.state_dict()
        nova_model_state = copy.deepcopy(model_state)
        coeff = 0.0
        for i, c in enumerate(self.clients):
            coeff = coeff + client_coeff[c.client_id] * client_n_data[c.client_id] / self.n_data
            for key in client_state[c.client_id]:
                if i == 0:
                    nova_model_state[key] = client_norm_grad[c.client_id][key] * client_n_data[c.client_id] / self.n_data
                else:
                    nova_model_state[key] = nova_model_state[key] + client_norm_grad[c.client_id][key] * \
                                            client_n_data[c.client_id] / self.n_data

        for key in model_state:
            model_state[key] -= coeff * nova_model_state[key]

        self.global_model.load_state_dict(model_state)

    # 模型评估
    def model_eval(self):
        self.global_model.eval()
        total_loss = 0.0
        correct = 0
        dataset_size = 0
        for batch_id, batch in enumerate(self.eval_loader):
            data, target = batch
            dataset_size += data.size()[0]

            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

            output = self.global_model(data)

            # sum up batch loss
            total_loss += torch.nn.functional.cross_entropy(output, target, reduction='sum').item()
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

        acc = 100.0 * (float(correct) / float(dataset_size))
        total_l = total_loss / dataset_size

        return acc, total_l


if __name__ == '__main__':
    arr = [1, 2, 3]
    for i, name in enumerate(arr):
        print(i, name)
