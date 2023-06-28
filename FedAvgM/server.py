import models, torch


class Server(object):

    def __init__(self, conf, eval_dataset):
        self.conf = conf

        # 服务器学习率
        self.lr = 0.1

        self.global_model = models.get_model(self.conf["model_name"])

        self.V = {}
        for name, params in self.global_model.state_dict().items():
            self.V[name] = torch.zeros_like(params)

        self.eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=self.conf["batch_size"], shuffle=True)

    def model_aggregate(self, weight_accumulator):

        for name in list(self.V.keys()):
            self.V[name] = (self.conf["beta"] * self.V[name] + weight_accumulator[name] * (1 / self.conf["clients"]))

        for name in list(weight_accumulator.keys()):
            if weight_accumulator[name].type() != self.V[name].type():
                weight_accumulator[name] += (self.conf["beta"] * self.V[name]).to(torch.int64)
            else:
                weight_accumulator[name] += self.conf["beta"] * self.V[name].float()

        for name, data in self.global_model.state_dict().items():
            update_per_layer = weight_accumulator[name]*self.lr
            if data.type() != update_per_layer.type():
                data.add_(update_per_layer.to(torch.int64))
            else:
                data.add_(update_per_layer)

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
