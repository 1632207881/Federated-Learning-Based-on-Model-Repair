import torch
import os
import numpy as np
import copy
import time
import random
from torch.utils.data import DataLoader
from readData import read_client_data_text, read_client_data

class serverBase(object):
    def __init__(self, args, times):
        self.device = args.device
        self.task = args.task
        self.dataset = args.dataset
        self.global_rounds = args.global_rounds
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.num_classes = args.num_classes
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.join_clients = int(self.num_clients * self.join_ratio)
        
        self.clients = []
        self.selected_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_acc = []
        self.rs_train_loss = []

        self.times = times
        self.eval_gap = args.eval_gap

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []

    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()
            
            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()
            
            for client in self.selected_clients:
                client.train()
            
            self.receive_models()
            self.aggregate_parameters()
            
            self.Budget.append(time.time() - s_t)
            print('-'*50, self.Budget[-1])

        print("\nBest global accuracy.")
        print(max(self.rs_test_acc))
        print(sum(self.Budget[1:])/len(self.Budget[1:]))
        print(f"acc:  {self.rs_test_acc}")
        print(f"loss: {self.rs_train_loss}")
        
    def set_clients(self, args, clientObj):
        for i in range(self.num_clients):
            if args.task == "NLP":
                train_data = read_client_data_text(self.dataset, i, is_train=True)
                test_data  = read_client_data_text(self.dataset, i, is_train=False)
            else:
                train_data = read_client_data(self.dataset, i, is_train=True)
                test_data  = read_client_data(self.dataset, i, is_train=False)

            client = clientObj(args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data))
            self.clients.append(client)

    def select_clients(self):
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))
        
        return selected_clients

    def send_models(self):
        for client in self.clients:
            client.set_parameters(self.global_model)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        tot_samples = 0
        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        for client in self.selected_clients:
            tot_samples += client.train_samples
            self.uploaded_weights.append(client.train_samples)
            self.uploaded_ids.append(client.id)
            self.uploaded_models.append(client.model)

        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()
        
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w
            
    def test_metrics(self):
        num_samples = []
        tot_correct = []
        for c in self.clients:
            ct, ns = c.test_metrics()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct

    def train_metrics(self):
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses
    
    def evaluate(self, acc=None, loss=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()
        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)
        
        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)
        
        print("Test Accurancy: {:.4f}".format(test_acc))
        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Std Test Accurancy: {:.4f}".format(np.std(accs)))      
    
    def evaluate_on_category(self):
        total_acc = []
        total_samples = []
        for client in self.clients:
            acc, samples = client.per_class_acc_for_global()
            total_acc.append(acc)
            total_samples.append(samples)
        total_acc = torch.tensor(total_acc)
        total_samples = torch.tensor(total_samples)
        acc = torch.sum(total_acc, dim=0) / torch.sum(total_samples, dim=0)
        return acc.tolist() 
    
    