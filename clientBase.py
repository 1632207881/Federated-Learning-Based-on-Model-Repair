import torch
import os
import numpy as np
import copy
import time
import random
from torch import nn
from torch.utils.data import DataLoader
from readData import read_client_data_text, read_client_data

class clientBase(object):
    def __init__(self, args, id, train_samples, test_samples):
        self.task = args.task
        self.model = copy.deepcopy(args.model)
        self.dataset = args.dataset
        self.device = args.device
        self.id = id

        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_steps = args.local_steps
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        # self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        #     optimizer=self.optimizer, 
        #     gamma=args.learning_rate_decay_gamma
        # )
        # self.learning_rate_decay = args.learning_rate_decay
    
    def set_parameters(self, model):
        # for new_param, old_param in zip(model.parameters(), self.model.parameters()):
        #     old_param.data = new_param.data.clone()
        self.model.load_state_dict(model.state_dict())
    
    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        
        if self.task == "NLP":
            train_data = read_client_data_text(self.dataset, self.id, is_train=True)
        else:
            train_data = read_client_data(self.dataset, self.id, is_train=True)
        
        return DataLoader(train_data, batch_size, drop_last=False, shuffle=False)
        
    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        
        if self.task == "NLP":
            test_data = read_client_data_text(self.dataset, self.id, is_train=False)
        else:
            test_data = read_client_data(self.dataset, self.id, is_train=False)


        return DataLoader(test_data, batch_size, drop_last=False, shuffle=False)
        
    def test_metrics(self):
        testloader = self.load_test_data()
        
        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []

        self.model.eval()
        with torch.no_grad():
            for x, y in testloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

        return test_acc, test_num

    def train_metrics(self):
        trainloader = self.load_train_data()
        
        train_num = 0
        losses = 0
        
        self.model.eval()
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]
        
        return losses, train_num

    # class-wise accuracy.
    def per_class_acc_for_global(self):
        testloader = self.load_test_data(batch_size=500)
        self.model.eval()
        
        label_acc =     [0. for _ in range(self.num_classes)]
        label_samples = [0. for _ in range(self.num_classes)]
        
        with torch.no_grad():
            for x, y in testloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                for output_value, y_value in zip(output, y):
                    if torch.argmax(output_value) == y_value:
                        label_acc[y_value]+=1
                    label_samples[y_value]+=1

        return label_acc, label_samples