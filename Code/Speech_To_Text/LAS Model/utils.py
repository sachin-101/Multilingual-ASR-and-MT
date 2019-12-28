import torch
import datetime
import numpy as np
from torch.nn.utils.rnn import pad_sequence


def train(model, device, train_loader, optimizer, epoch, print_interval, writer=None, log_interval=-1):
    
    model.train()
    print(f'Training, Logging: Mean loss of previous {print_interval} batches \n')
    
    running_loss = []
    date1 = datetime.datetime.now()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        loss, _ = model(data, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss.append(loss.detach().item())    # update running loss
        
        # writing to console after print_interval batches
        if (batch_idx+1) % print_interval == 0:
            date2 = datetime.datetime.now()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tMean Loss : {:.6f}\t time {}:'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), 
                np.mean(running_loss[-print_interval:]), 
                date2 - date1))
            
            date1 = date2

        # Writing to tensorboard
        if (batch_idx+1) % log_interval == 0:
            if writer:
                global_step = epoch * len(train_loader) + batch_idx
                writer.add_scalar('Loss', np.mean(running_loss[-log_interval:]), global_step)

        
# def test(model, device, test_loader, log_interval):
#     model.eval()
#     loss = 0
#     print('-'*10, '\nTesting')
#     date1 = datetime.datetime.now()
#     for batch_idx, (data, target, in_len, tgt_len) in enumerate(test_loader):
#         data, target = data.to(device), target.to(device)
#         in_len, tgt_len = in_len.to(device), tgt_len.to(device)
#         output = model(data)
#         loss += F.ctc_loss(output, target, in_len, tgt_len).detach().item()  # default blank token : 0
#         if batch_idx % log_interval == 0:
#             date2 = datetime.datetime.now()
#             print('Train Epoch: [{}/{} ({:.0f}%)]\t time {}:'.format(
#                 batch_idx * len(data), len(test_loader.dataset),
#                 100. * batch_idx / len(test_loader), date2 - date1))
#             date1 = date2
#     return loss/len(test_loader)