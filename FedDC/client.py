from torch.utils.data import DataLoader, sampler, Subset
import torch

import models


class Client(object):

    def __init__(self, conf, model, train_dataset, non_iid, id=-1):
        self.client_id = id

        self.conf = conf
        self.local_model = models.get_model(self.conf["model_name"])
        self.local_model.load_state_dict(model.state_dict())

        self.last_local_parameter = None
        for param in self.local_model.parameters():
            if not isinstance(self.last_local_parameter, torch.Tensor):
                self.last_local_parameter = param.reshape(-1)
            else:
                self.last_local_parameter = torch.cat((self.last_local_parameter, param.reshape(-1)), 0)

        self.hist = 0.0

        self.last_local_update = 0.0

        sub_trainset: Subset = Subset(train_dataset, indices=non_iid)

        self.train_loader = DataLoader(sub_trainset, batch_size=conf["batch_size"], shuffle=False)

    def get_hist(self, global_parameter):
        local_parameter = None
        for param in self.local_model.parameters():
            if not isinstance(local_parameter, torch.Tensor):
                local_parameter = param.reshape(-1)
            else:
                local_parameter = torch.cat((local_parameter, param.reshape(-1)), 0)

        self.hist = self.hist + (local_parameter - global_parameter)

    def get_last_local_update(self):
        _parameter = None
        for param in self.local_model.parameters():
            if not isinstance(_parameter, torch.Tensor):
                _parameter = param.reshape(-1)
            else:
                _parameter = torch.cat((_parameter, param.reshape(-1)), 0)

        self.last_local_update = _parameter - self.last_local_parameter

        self.last_local_parameter = _parameter

    def local_train(self, global_model, last_global_update):

        global_model_param = None
        for param in global_model.parameters():
            if not isinstance(global_model_param, torch.Tensor):
                global_model_param = param.reshape(-1)
            else:
                global_model_param = torch.cat((global_model_param, param.reshape(-1)), 0)

        state_update_diff = self.last_local_update - last_global_update

        for name, param in global_model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())

        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.conf['lr'], momentum=self.conf['momentum'])
        self.local_model.train()
        for e in range(self.conf["local_epochs"]):

            for batch_id, batch in enumerate(self.train_loader):
                data, target = batch
                if torch.cuda.is_available():
                    data = data.cuda()
                    target = target.cuda()

                optimizer.zero_grad()
                output = self.local_model(data)

                if len(target) == 1:
                    _output = torch.zeros(1, len(output))
                    output = _output

                local_parameter = None
                for param in self.local_model.parameters():
                    if not isinstance(local_parameter, torch.Tensor):
                        local_parameter = param.reshape(-1)
                    else:
                        local_parameter = torch.cat((local_parameter, param.reshape(-1)), 0)

                loss_cp = self.conf["alpha"] / 2 * torch.sum(
                    (local_parameter - (global_model_param - self.hist)) * (
                            local_parameter - (global_model_param - self.hist)))

                loss_cg = torch.sum(local_parameter * state_update_diff)

                loss = torch.nn.functional.cross_entropy(output, target)
                loss = loss + loss_cp + loss_cg
                loss.backward()
                optimizer.step()

            print("Client {} Epoch {} done.".format(self.client_id, e))

        self.get_hist(global_model_param)
        self.get_last_local_update()

        diff = dict()
        for name, data in self.local_model.state_dict().items():
            diff[name] = (data - global_model.state_dict()[name])
        return diff
