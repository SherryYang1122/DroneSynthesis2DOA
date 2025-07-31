from torch.utils.data import DataLoader, Dataset
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau




def create_data_loader(train_data, batch_size, device, num_gpu):
    num_workers = 2 * num_gpu
    # Create the truncated dataset
    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return dataloader


class TruncatedDataset(Dataset):
    def __init__(self, original_dataset, max_length):
        self.original_dataset = original_dataset
        self.max_length = max_length
        self.samples = self.truncate_samples()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def truncate_samples(self):
        truncated_samples = []
        for sample in self.original_dataset:
            label = sample[1]  # Assume the first element of the sample is the label
            data = sample[0]   # Assume the second element of the sample is the data
            # Truncate the data if its length exceeds the maximum length
            if len(data) >= self.max_length:
                num_segments = len(data) // self.max_length
                for i in range(num_segments):
                    truncated_data = data[i*self.max_length : (i+1)*self.max_length]
                    truncated_samples.append((truncated_data, label[i*self.max_length : (i+1)*self.max_length]))
        return truncated_samples

def train(model, train_dataloader, val_dataloader, loss_fn, optimiser, device, epochs, writer):

    loss_list = []
    loss_val_list = []
    scheduler = ReduceLROnPlateau(optimiser, mode='min', patience=5, factor=0.5, verbose=True) #0.1
    # Variables for Early Stopping logic
    patienceEarlyStop = 30 #20
    counter = 0
    best_loss = None

    for i in range(epochs):
        print(f"Epoch {i + 1}")

        loss, acc = train_single_epoch(model, train_dataloader, loss_fn, optimiser, device)
        loss_list.append(loss)
        writer.add_scalar("Loss/train", loss, i)
        writer.add_scalar("Accuracy/train", acc, i)

        loss_val, acc_val = validate_single_epoch(model, val_dataloader, loss_fn, optimiser, scheduler, device)
        loss_val_list.append(loss_val)
        writer.add_scalar("Loss/validation", loss_val, i)
        writer.add_scalar("Accuracy/validatio", acc_val, i)

        print("---------------------------")
        # Early Stopping logic
        if best_loss is None:
            best_loss = loss_val   # Initialize best loss
        elif loss_val > best_loss:
            counter += 1           # Increment counter if no improvement
            if counter >= patienceEarlyStop:
                print("Early stopping")
                break              # Stop training if patience is exceeded
        else:
            best_loss = loss_val   # Update best loss
            counter = 0            # Reset counter if improvement
    print("Finished training")
    writer.flush()

    return loss_list, loss_val_list


def train_single_epoch(model, data_loader, loss_fn, optimiser, device):

    model.train()
    loss_avg = []
    frame_num = 0
    correct_train = 0

    #for inputs, target, _ in data_loader:
    if isinstance(loss_fn, nn.CrossEntropyLoss):
        for input, target in data_loader:
            input, target = input.to(device), target.to(device)
            prediction = model(input)
            #loss = loss_fn(prediction, labels)
            loss = loss_fn(prediction, target)
            #loss_avg.append(loss)
            loss_avg.append(loss.item()*len(target))
            frame_num = frame_num + len(target)
            # backpropagate error and update weights
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            _, predicted = torch.max(prediction, 1)
            correct_train += (predicted == target).sum().item()
        loss = torch.sum(torch.tensor(loss_avg, device=device))/frame_num
        train_acc = 100 * correct_train / frame_num
        print(f"train loss: {loss.item()}; train acc: {train_acc}%")
    else:
        for item in data_loader:
            if len(item) == 4:
                input, mic_pairs, target, batch_lengths = item
                mic_pairs = [pair for pair in mic_pairs]
                mic_pairs = torch.stack(mic_pairs)
                input, mic_pairs, target = input.to(device), mic_pairs.to(device), target.to(device)
                prediction, _ = model(input, mic_pairs, batch_lengths)
            else:
                input, target, batch_lengths = item
                input, target = input.to(device), target.to(device)
                prediction, _ = model(input, batch_lengths)
            loss = loss_fn(prediction, target)
            loss_avg.append(loss.item()*target.shape[0]*target.shape[1])
            frame_num = frame_num + torch.sum(batch_lengths)
            # backpropagate error and update weights
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        loss = torch.sum(torch.tensor(loss_avg, device=device))/frame_num
        train_acc = loss.item()
        print(f"train loss: {loss.item()}")
    return loss.item(), train_acc


def validate_single_epoch(model, data_loader, loss_fn, optimiser, scheduler, device):
    model.eval()

    loss_avg = []
    
    frame_num = 0
    correct_val = 0
    with torch.no_grad():
        if isinstance(loss_fn, nn.CrossEntropyLoss):
            for input, target in data_loader:
                input, target = input.to(device), target.to(device)
                prediction = model(input)
                loss = loss_fn(prediction, target)
                loss_avg.append(loss.item()*len(target))
                frame_num = frame_num + len(target)
                _, predicted = torch.max(prediction, 1)
                correct_val += (predicted == target).sum().item()
            loss = torch.sum(torch.tensor(loss_avg, device=device))/frame_num
            scheduler.step(loss)
            val_acc = 100 * correct_val / frame_num
            print(f"val loss:  {loss.item()}; val acc: {val_acc}%")
            print(f"lr:  {optimiser.param_groups[0]['lr']}")
        else:
            for item in data_loader:
                if len(item) == 4:
                    input, mic_pairs, target, batch_lengths = item
                    mic_pairs = [pair for pair in mic_pairs]
                    mic_pairs = torch.stack(mic_pairs)
                    input, mic_pairs, target = input.to(device), mic_pairs.to(device), target.to(device)
                    prediction, _ = model(input, mic_pairs, batch_lengths)
                else:
                    input, target, batch_lengths = item
                    input, target = input.to(device), target.to(device)
                    prediction, _ = model(input, batch_lengths)
                loss = loss_fn(prediction, target)
                loss_avg.append(loss.item()*target.shape[0]*target.shape[1])
                frame_num = frame_num + torch.sum(batch_lengths)
            loss = torch.sum(torch.tensor(loss_avg, device=device))/frame_num
            scheduler.step(loss)
            val_acc = loss.item()
            print(f"val loss:  {loss.item()}")
            print(f"lr:  {optimiser.param_groups[0]['lr']}")            
 
    return loss.item(), val_acc



