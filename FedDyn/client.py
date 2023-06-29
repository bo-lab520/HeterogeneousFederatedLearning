
from torch.utils.data import DataLoader, sampler, Subset
import torch

import models

class Client(object):

    def __init__(self, conf, model, train_dataset, non_iid, id=-1):
        self.client_id = id

        self.conf = conf
        self.local_model = models.get_model(self.conf["model_name"])
        self.local_model.load_state_dict(model.state_dict())



        sub_trainset: Subset = Subset(train_dataset, indices=non_iid)

        self.train_loader = DataLoader(train_dataset, batch_size=conf["batch_size"], shuffle=False)

    def local_train(self, global_model, avg_mdl_param, local_grad_vector):
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

                loss_f = torch.nn.functional.cross_entropy(output, target)

                local_par_list = None
                for param in self.local_model.parameters():
                    if not isinstance(local_par_list, torch.Tensor):
                        local_par_list = param.reshape(-1)
                    else:
                        local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)

                loss_algo = self.conf["alpha"] * torch.sum(local_par_list * (-avg_mdl_param + local_grad_vector))

                loss = loss_f + loss_algo

                loss.backward()
                optimizer.step()

            print("Client {} Epoch {} done.".format(self.client_id, e))

        diff = dict()
        for name, data in self.local_model.state_dict().items():
            diff[name] = (data - global_model.state_dict()[name])
        # return diff
        return self.local_model, diff