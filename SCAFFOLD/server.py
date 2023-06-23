import models, torch


class Server(object):

    def __init__(self, conf, eval_dataset):
        self.conf = conf

        self.global_model = models.get_model(self.conf["model_name"])
        self.control_global = models.get_model(self.conf["model_name"])
        for name, data in self.control_global.state_dict().items():
            data.add_(-data)

        self.eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=self.conf["batch_size"], shuffle=True)

    def model_aggregate(self, delta_x):
        for name, data in self.global_model.state_dict().items():
            update_per_layer = delta_x[name] * (1 / self.conf["clients"])
            if data.type() != update_per_layer.type():
                data.add_(update_per_layer.to(torch.int64))
            else:
                data.add_(update_per_layer)

    def control_aggregate(self, delta_c):
        for name, data in self.control_global.state_dict().items():
            update_per_layer = delta_c[name] * (1 / self.conf["clients"])
            if data.type() != update_per_layer.type():
                data.add_(update_per_layer.to(torch.int64))
            else:
                data.add_(update_per_layer)

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

            total_loss += torch.nn.functional.cross_entropy(output, target, reduction='sum').item()
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

        acc = 100.0 * (float(correct) / float(dataset_size))
        total_l = total_loss / dataset_size

        return acc, total_l
