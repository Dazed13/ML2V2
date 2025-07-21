import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision
import torchvision.transforms as transforms
import numpy as np
import copy
import random
from collections import OrderedDict
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CNN Model for Image Classification
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
        self.fc2 = nn.Linear(512, 6)  # Only 6 animal classes
        self.dropout = nn.Dropout(0.5)  # Increased dropout rate
    
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Client class to simulate mobile devices
class Client:
    def __init__(self, client_id, data, battery_power, comm_strength, computing_power):
        self.client_id = client_id
        self.data = data  # Dataset assigned to this client
        self.battery_power = battery_power  # Range 0-100
        self.comm_strength = comm_strength  # Range 0-100
        self.computing_power = computing_power  # Range 0-100
        self.model = None
        
        # Validate data
        if len(self.data) == 0:
            raise ValueError(f"Client {client_id} has no data!")
    
    def receive_model(self, model):
        """Receive the global model from the server"""
        self.model = copy.deepcopy(model)
        self.model.to(device)
    
    def update_model(self, batch_size, epochs, learning_rate):
        """Update the model using local data"""
        if len(self.data) == 0:
            raise ValueError(f"Client {self.client_id} has no data to train on!")
            
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
        
        # Create dataloaders
        train_loader = DataLoader(self.data, batch_size=min(batch_size, len(self.data)), shuffle=True)
        
        # Train the model
        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward + backward + optimize
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                # Properly detach the loss before converting to scalar
                running_loss += loss.detach().item()
            
            scheduler.step()
        
        return self.model.state_dict()

# Server class to orchestrate federated learning
class FederatedServer:
    def __init__(self, model, clients, client_fraction=0.1):
        self.model = model
        self.clients = clients
        self.client_fraction = client_fraction  # Fraction of clients to select each round
        self.global_model = model.state_dict()
    
    def select_clients(self):
        """Select clients based on battery power, communication strength, and computing capabilities"""
        # Calculate selection probability based on client parameters
        selection_scores = []
        for client in self.clients:
            # Higher scores mean higher chances of selection
            score = (client.battery_power + client.comm_strength + client.computing_power) / 3.0
            selection_scores.append(score)
        
        # Normalize scores to get selection probabilities
        total_score = sum(selection_scores)
        selection_probs = [score / total_score for score in selection_scores]
        
        # Determine number of clients to select
        num_clients = max(int(self.client_fraction * len(self.clients)), 1)
        
        # Select clients
        selected_clients = np.random.choice(
            self.clients, 
            size=num_clients, 
            replace=False, 
            p=selection_probs
        )
        
        return selected_clients
    
    def distribute_model(self, selected_clients):
        """Send the global model to selected clients"""
        for client in selected_clients:
            client.receive_model(copy.deepcopy(self.model))
    
    def aggregate_updates(self, client_updates, client_data_sizes):
        """Aggregate model updates from clients using FedAvg algorithm"""
        # Initialize the global model with zeros
        global_dict = OrderedDict()
        total_size = sum(client_data_sizes)
        
        # Initialize with zeros
        for key in client_updates[0].keys():
            global_dict[key] = torch.zeros_like(client_updates[0][key], dtype=torch.float32)
        
        # Weighted average of client models
        for i, client_update in enumerate(client_updates):
            weight = client_data_sizes[i] / total_size
            for key in global_dict.keys():
                # Convert to float32 before multiplication
                update = client_update[key].float()
                global_dict[key] += update * weight
        
        self.global_model = global_dict
        self.model.load_state_dict(self.global_model)
    
    def train(self, rounds, local_batch_size, local_epochs, learning_rate):
        """Run federated learning for specified number of rounds"""
        accuracy_history = []
        
        for round_num in range(1, rounds + 1):
            print(f"Round {round_num}/{rounds}")
            
            # Select clients
            selected_clients = self.select_clients()
            print(f"Selected {len(selected_clients)} clients")
            
            # Distribute global model to selected clients
            self.distribute_model(selected_clients)
            
            # Collect updated models and their data sizes
            client_updates = []
            client_data_sizes = []
            
            for client in selected_clients:
                # Client performs local training
                updated_model = client.update_model(local_batch_size, local_epochs, learning_rate)
                client_updates.append(updated_model)
                client_data_sizes.append(len(client.data))
            
            # Aggregate updates
            self.aggregate_updates(client_updates, client_data_sizes)
            
            # Evaluate the global model on test data
            accuracy = self.evaluate()
            accuracy_history.append(accuracy)
            print(f"Round {round_num} accuracy: {accuracy:.4f}")
        
        return accuracy_history
    
    def evaluate(self, test_loader):
        """Evaluate the global model on test data"""
        
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

# Filter dataset to include only animal classes
def filter_animal_classes(dataset):
    """
    Filter CIFAR-10 dataset to include only animal classes:
    - bird (2), cat (3), deer (4), dog (5), frog (6), horse (7)
    Excludes: airplane (0), automobile (1), ship (8), truck (9)
    """
    animal_classes = [2, 3, 4, 5, 6, 7]  # Indices of animal classes in CIFAR-10
    animal_indices = [i for i, (_, label) in enumerate(dataset) if label in animal_classes]
    
    # Create a new dataset with only animal classes
    animal_dataset = torch.utils.data.Subset(dataset, animal_indices)
    
    # Create a mapping from old label to new sequential label (0-5)
    label_mapping = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5}
    
    # Create a custom dataset with remapped labels
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

# Function to simulate clients with non-IID data distribution using Dirichlet distribution
def create_non_iid_clients(dataset, num_clients=1000):
    """
    Create clients with non-IID data distribution using Dirichlet distribution
    Each client gets data with a preference for certain classes
    """
    # Get all targets
    targets = torch.tensor([dataset[i][1] for i in range(len(dataset))])
    num_classes = 6  # Number of animal classes
    
    # Initialize client indices
    client_indices = [[] for _ in range(num_clients)]
    
    # For each class, distribute indices among clients according to Dirichlet distribution
    for k in range(num_classes):
        # Get indices of samples in this class
        idx_k = torch.where(targets == k)[0].tolist()
        
        # Sample from Dirichlet distribution
        proportions = np.random.dirichlet(np.repeat(0.5, num_clients))  # alpha=0.5 for moderate non-IID
        
        # Ensure minimum samples per client (at least 1 sample per class)
        min_samples = 1
        proportions = np.maximum(proportions, min_samples / len(idx_k))
        proportions = proportions / proportions.sum()  # Renormalize
        
        # Distribute indices according to proportions
        proportions = np.array([p/sum(proportions) * len(idx_k) for p in proportions])
        proportions = proportions.astype(int)
        
        # Adjust last value to account for rounding errors
        proportions[-1] = len(idx_k) - sum(proportions[:-1])
        
        # Distribute indices
        indices = np.split(np.random.permutation(idx_k), np.cumsum(proportions)[:-1])
        
        # Add indices to client datasets
        for i, idcs in enumerate(indices):
            client_indices[i].extend(idcs.tolist())
    
    # Create client objects with varying capabilities
    clients = []
    for i in range(num_clients):
        # Skip clients with no data
        if len(client_indices[i]) == 0:
            continue
            
        client_data = torch.utils.data.Subset(dataset, client_indices[i])
        
        # Simulate different device capabilities
        battery_power = np.random.uniform(30, 100)  # Min battery level is 30%
        comm_strength = np.random.uniform(20, 100)  # Communication strength varies
        computing_power = np.random.uniform(40, 100)  # Computing power varies
        
        client = Client(i, client_data, battery_power, comm_strength, computing_power)
        clients.append(client)
    
    return clients

# Main function to run the federated learning simulation
def main():
    # Load CIFAR-10 dataset
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
    
    # Load the original datasets
    original_train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    
    original_test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    # Filter and remap to only include animal classes
    train_dataset = filter_animal_classes(original_train_dataset)
    test_dataset = filter_animal_classes(original_test_dataset)
    
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
    
    # Print dataset information
    print(f"Original training dataset size: {len(original_train_dataset)}")
    print(f"Animal-only training dataset size: {len(train_dataset)}")
    print(f"Original test dataset size: {len(original_test_dataset)}")
    print(f"Animal-only test dataset size: {len(test_dataset)}")
    
    # Create a CNN model
    model = CNN().to(device)
    
    # Create clients with non-IID data
    num_clients = 1000  # Simulating 100 clients instead of 1000 for performance
    clients = create_non_iid_clients(train_dataset, num_clients)
    
    # Run federated learning
    federated_rounds = 50
    local_batch_size = 32
    local_epochs = 10  # Increased from 5 to 10
    learning_rate = 0.001  # Reduced from 0.01 to 0.001 for Adam optimizer
    
    # Initialize the federated learning server with more clients
    server = FederatedServer(model, clients, client_fraction=0.2)  # Increased from 0.1 to 0.2
    
    # Make sure the test_loader is passed to the server
    server.evaluate = lambda: FederatedServer.evaluate(server, test_loader)
    
    accuracy_history = server.train(
        federated_rounds,
        local_batch_size,
        local_epochs,
        learning_rate
    )
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, federated_rounds + 1), accuracy_history)
    plt.xlabel('Federated Learning Rounds')
    plt.ylabel('Test Accuracy')
    plt.title('Federated Learning Performance')
    plt.grid(True)
    plt.savefig('federated_learning_results.png')
    plt.show()
    
    print(f"Final model accuracy: {accuracy_history[-1]:.4f}")
    
    # Save the model
    torch.save(server.model.state_dict(), 'federated_model.pth')

if __name__ == "__main__":
    main()