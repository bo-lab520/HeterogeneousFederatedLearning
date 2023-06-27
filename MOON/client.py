import time

from torch.utils.data import DataLoader, sampler
import torch

import get_model


class Client(object):

    def __init__(self, conf, model, train_dataset, non_iid, id=-1):
        self.client_id = id

        self.conf = conf
        self.local_model = get_model.MOONModel(self.conf["model_name"], self.conf["out_dim"], self.conf["classes"])
        self.local_model.load_state_dict(model.state_dict())

        self.last_epoch_model = self.local_model

        self.train_dataset = train_dataset

        self.non_iid = non_iid

        # 按ID对训练集合的拆分
        # all_range = list(range(len(self.train_dataset)))
        # data_len = int(len(self.train_dataset) / self.conf['clients'])
        # train_indices = all_range[(int(id) - 1) * data_len: int(id) * data_len]

        self.train_loader = DataLoader(self.train_dataset, batch_size=conf["batch_size"], shuffle=True)

    def local_train(self, global_model):
        for name, param in global_model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())

        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.conf['lr'], momentum=self.conf['momentum'])
        cos = torch.nn.CosineSimilarity(dim=-1)
        self.local_model.train()

        for e in range(self.conf["local_epochs"]):
            _data = torch.zeros(self.conf["batch_size"], self.conf["channels"], self.conf["pic_size"],
                                self.conf["pic_size"])
            _target = torch.zeros(self.conf["batch_size"], dtype=torch.long)
            index = 0
            for batch_id, batch in enumerate(self.train_loader):
                data, target = batch
                # non-iid data
                for i in range(len(target)):
                    if int(target[i]) in self.non_iid:
                        _target[index] = target[i]
                        _data[index] = data[i]
                        index += 1
                        if index == self.conf["batch_size"]:
                            index = 0
                            if torch.cuda.is_available():
                                _data = data.cuda()
                                _target = target.cuda()

                            optimizer.zero_grad()
                            _, cur_proj, output = self.local_model(_data)
                            _, global_proj, _ = global_model(_data)
                            # 计算本地模型与全局模型的representation的相似度
                            posi = cos(cur_proj, global_proj)
                            logits = posi.reshape(-1, 1)
                            # 计算当前轮次的本地模型与上一轮次的本地模型的表示特征相似性
                            # self.last_epoch_model.cuda()

                            _, last_proj, _ = self.last_epoch_model(_data)
                            nega = cos(cur_proj, last_proj)
                            logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)
                            # self.last_epoch_model.to('cpu')
                            logits /= self.conf["T"]
                            # labels = torch.zeros(data.size(0)).cuda().long()
                            labels = torch.zeros(_data.size(0)).cpu().long()
                            loss2 = self.conf["mu"] * torch.nn.functional.cross_entropy(logits, labels)

                            loss1 = torch.nn.functional.cross_entropy(output, _target)
                            loss = loss1 + loss2
                            loss.backward()
                            optimizer.step()

            print("Client {} Epoch {} done.".format(self.client_id, e))

        # 更新上一轮本地模型
        self.last_epoch_model = self.local_model

        diff = dict()
        for name, data in self.local_model.state_dict().items():
            diff[name] = (data - global_model.state_dict()[name])
        return diff
