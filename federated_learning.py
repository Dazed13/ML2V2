import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import copy
import random
from collections import OrderedDict
import matplotlib.pyplot as plt
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 6)  
        self.dropout = nn.Dropout(0.5) 
    
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

#client class. multiple of these are instantiated in each round so that we can effectively simulate passing the model to clients as we do in federated learning.
class Client:
    def __init__(self, client_id, data, battery_power, comm_strength, computing_power):
        self.client_id = client_id
        self.full_data = data 
        self.data = None  
        self.battery_power = battery_power  
        self.comm_strength = comm_strength  
        self.computing_power = computing_power 
        self.model = None
        self.criterion = nn.CrossEntropyLoss()
        
    def update_attributes(self):
        self.battery_power = np.random.uniform(30, 100)  
        self.comm_strength = np.random.uniform(20, 100)  
        self.computing_power = np.random.uniform(40, 100)
        
    def set_data_usage(self, usage_ratio):
        if usage_ratio < 0.0 or usage_ratio > 1.0:
            raise ValueError("Usage ratio must be between 0.0 and 1.0")
            
        num_samples = max(1, int(len(self.full_data) * usage_ratio))
        
        if usage_ratio >= 0.99:
            self.data = self.full_data
        else:
            indices = list(range(len(self.full_data)))
            random.shuffle(indices) 
            selected_indices = indices[:num_samples]
            self.data = torch.utils.data.Subset(self.full_data, selected_indices)
    
    def receive_model(self, model):

        self.model = copy.deepcopy(model)
        self.model.to(device)
    
    def update_model(self, batch_size, epochs, learning_rate):

            
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
        
        train_loader = DataLoader(self.data, batch_size=min(batch_size, len(self.data)), shuffle=True)
        
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                epoch_loss += loss.detach().item()
            scheduler.step()
            total_loss += epoch_loss
            num_batches += len(train_loader)
        avg_loss = total_loss / num_batches
        return self.model.state_dict(), avg_loss

class FederatedServer:
    def __init__(self, model, clients, client_fraction, data_growth_rate):
        self.model = model
        self.clients = clients
        self.client_fraction = client_fraction
        self.global_model = model.state_dict()
        self.criterion = nn.CrossEntropyLoss()
        self.data_growth_rate = data_growth_rate 
        self.current_round = 0  
    
    def select_clients(self):

        selection_scores = []
        for client in self.clients:
            score = (client.battery_power + client.comm_strength + client.computing_power) / 3.0
            selection_scores.append(score)
        

        total_score = sum(selection_scores)
        selection_probs = [score / total_score for score in selection_scores]
        
        num_clients = max(int(self.client_fraction * len(self.clients)), 1)
        
        #not hard assignment, but probabilistic
        selected_clients = np.random.choice(
            self.clients, 
            size=num_clients, 
            replace=False, 
            p=selection_probs
        )
        
        return selected_clients
    
    def distribute_model(self, selected_clients):
        for client in selected_clients:
            client.receive_model(copy.deepcopy(self.model))
    
    def aggregate_updates(self, client_updates, client_data_sizes):

        global_dict = OrderedDict()
        total_size = sum(client_data_sizes)

        for key in client_updates[0].keys():
            global_dict[key] = torch.zeros_like(client_updates[0][key], dtype=torch.float32)
        
        # average weight of client updates
        for i, client_update in enumerate(client_updates):
            weight = client_data_sizes[i] / total_size
            for key in global_dict.keys():
                update = client_update[key].float()
                global_dict[key] += update * weight
        
        self.global_model = global_dict
        self.model.load_state_dict(self.global_model)
    
    def train(self, rounds, local_batch_size, local_epochs, learning_rate):
        accuracy_history = []
        loss_history = []
        selected_clients_history = []  # Track selected clients for each round
        
        initial_usage = 0.8
        for client in self.clients:
            client.set_data_usage(initial_usage)
        
        for round_num in range(1, rounds + 1):
            self.current_round = round_num
            print("Round ", round_num)
            
            current_usage = min(1.0, initial_usage + (round_num - 1) * self.data_growth_rate)
            
            for client in self.clients:
                client.set_data_usage(current_usage)
                #client.update_attributes()  
            
            selected_clients = self.select_clients()
            selected_clients_history.append(selected_clients)
            print(f"Selected {len(selected_clients)} clients")
            
            self.distribute_model(selected_clients)
            
            client_updates = []
            client_data_sizes = []
            round_loss = 0.0
            
            for client in selected_clients:
                updated_model, client_loss = client.update_model(local_batch_size, local_epochs, learning_rate)
                client_updates.append(updated_model)
                client_data_sizes.append(len(client.data))
                round_loss += client_loss
            
            avg_round_loss = round_loss / len(selected_clients)
            loss_history.append(avg_round_loss)
            
            self.aggregate_updates(client_updates, client_data_sizes)
            
            accuracy = self.evaluate()
            accuracy_history.append(accuracy)
            print(f"Round {round_num} - Loss: {avg_round_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        return accuracy_history, loss_history, selected_clients_history
    
    def evaluate(self, test_loader):
        
        self.model.eval()
        self.model.to(device)
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        return accuracy

def filter_animal_classes(dataset):
    #the dataset has 4 other non animal classes, but these are filtered out
    animal_classes = [2, 3, 4, 5, 6, 7]
    animal_indices = [i for i, (_, label) in enumerate(dataset) if label in animal_classes]

    animal_dataset = torch.utils.data.Subset(dataset, animal_indices)
    
    label_mapping = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5}
    
    class RemappedDataset(Dataset):
        def __init__(self, subset, mapping):
            self.subset = subset
            self.mapping = mapping
            
        def __getitem__(self, idx):
            img, old_label = self.subset[idx]
            new_label = self.mapping[old_label]
            return img, new_label
            
        def __len__(self):
            return len(self.subset)
    
    return RemappedDataset(animal_dataset, label_mapping)

def create_non_iid_clients(dataset, num_clients=1000):

    targets = torch.tensor([dataset[i][1] for i in range(len(dataset))])
    num_classes = 6  

    client_indices = [[] for _ in range(num_clients)]
    
    for k in range(num_classes):
        idx_k = torch.where(targets == k)[0].tolist()
        
        proportions = np.random.dirichlet(np.repeat(0.5, num_clients))  

        min_samples = 1
        proportions = np.maximum(proportions, min_samples / len(idx_k))
        proportions = proportions / proportions.sum() 

        proportions = np.array([p/sum(proportions) * len(idx_k) for p in proportions])
        proportions = proportions.astype(int)

        proportions[-1] = len(idx_k) - sum(proportions[:-1])

        indices = np.split(np.random.permutation(idx_k), np.cumsum(proportions)[:-1])
        
        for i, idcs in enumerate(indices):
            client_indices[i].extend(idcs.tolist())
    

    clients = []
    for i in range(num_clients):
        if len(client_indices[i]) == 0:
            continue
            
        client_data = torch.utils.data.Subset(dataset, client_indices[i])
        #random initialization of client attributes
        battery_power = np.random.uniform(30, 100)  
        comm_strength = np.random.uniform(20, 100)  
        computing_power = np.random.uniform(40, 100)
        
        client = Client(i, client_data, battery_power, comm_strength, computing_power)
        clients.append(client)
    
    return clients

def train_centralized(model, train_loader, test_loader, epochs, learning_rate):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
    
    train_losses = []
    test_accuracies = []
    
    for epoch in range(epochs):

        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.detach().item()
        
        scheduler.step()
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        test_accuracies.append(accuracy)
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}')
    
    return train_losses, test_accuracies

#plot function
def plot_results(centralized_results, federated_results, selected_clients_history):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(centralized_results['test_acc']) + 1), 
             centralized_results['test_acc'], 
             label='Centralized')
    plt.plot(range(1, len(federated_results['test_acc']) + 1), 
             federated_results['test_acc'], 
             label='Federated')
    plt.xlabel('Epochs/Rounds')
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy Comparison')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    rounds = range(1, len(federated_results['test_acc']) + 1)
    
    cumulative_cost = []
    total_cost = 0
    for round_clients in selected_clients_history:
        round_cost = sum(100 - client.comm_strength for client in round_clients)
        total_cost += round_cost
        cumulative_cost.append(total_cost)
    
    plt.plot(rounds, cumulative_cost, 'r-', label='Cumulative Cost')
    plt.xlabel('Federated Learning Rounds')
    plt.ylabel('Communication Cost (100 - Communication Strength)')
    plt.title('Communication Cost Based on Client Communication Strength')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('learning_comparison.png')

def main():

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    

    original_train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    
    original_test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    train_dataset = filter_animal_classes(original_train_dataset)
    test_dataset = filter_animal_classes(original_test_dataset)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

    
    print("\nCentralized model.")
    centralized_model = CNN().to(device)
    centralized_results = {}
    centralized_results['train_loss'], centralized_results['test_acc'] = train_centralized(
        centralized_model, train_loader, test_loader, epochs=30, learning_rate=0.001
    )
    

    print("\nFederated learning.")
    federated_model = CNN().to(device)
    num_clients = 1000
    clients = create_non_iid_clients(train_dataset, num_clients)
    server = FederatedServer(
        federated_model, 
        clients, 
        client_fraction=0.2,
        data_growth_rate=0.02
    )
    
    

    federated_rounds = 30
    local_batch_size = 32
    local_epochs = 10
    learning_rate = 0.001
    
    server.evaluate = lambda: FederatedServer.evaluate(server, test_loader)
    
    federated_results = {}
    federated_results['test_acc'], federated_results['train_loss'], selected_clients_history = server.train(
        federated_rounds,
        local_batch_size,
        local_epochs,
        learning_rate
    )

    plot_results(centralized_results, federated_results, selected_clients_history)
    
    print(f"\nFinal centralized model accuracy: {centralized_results['test_acc'][-1]:.4f}")
    print(f"Final federated model accuracy: {federated_results['test_acc'][-1]:.4f}")

    torch.save(centralized_model.state_dict(), 'centralized_model.pth')
    torch.save(server.model.state_dict(), 'federated_model.pth')

if __name__ == "__main__":
    main()