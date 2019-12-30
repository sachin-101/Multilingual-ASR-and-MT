import datetime
import torch
import torch.nn.functional as F


def train(model, device, train_loader, optimizer, epoch, log_interval):
    model.train()
    print('Training')
    date1 = datetime.datetime.now()
    for batch_idx, (data, target, in_len, tgt_len) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        in_len, tgt_len = in_len.to(device), tgt_len.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.ctc_loss(output, target, in_len, tgt_len)  # default blank token : 0
        loss.backward()
        optimizer.step()    
        if batch_idx % log_interval == 0:
            date2 = datetime.datetime.now()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t time {}:'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), date2 - date1))
            date1 = date2


def test(model, device, test_loader, log_interval):
    model.eval()
    loss = 0
    print('-'*10, '\nTesting')
    date1 = datetime.datetime.now()
    for batch_idx, (data, target, in_len, tgt_len) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        in_len, tgt_len = in_len.to(device), tgt_len.to(device)
        output = model(data)
        loss += F.ctc_loss(output, target, in_len, tgt_len).detach().item()  # default blank token : 0
        if batch_idx % log_interval == 0:
            date2 = datetime.datetime.now()
            print('Train Epoch: [{}/{} ({:.0f}%)]\t time {}:'.format(
                batch_idx * len(data), len(test_loader.dataset),
                100. * batch_idx / len(test_loader), date2 - date1))
            date1 = date2
    return loss/len(test_loader)



def check_on_personal(model, device, idx, dataset, token_to_char):
    x, target, _, _ = dataset.__getitem__(idx) 
    output = model(x[None,:,:,:].to(device))
    sent = []
    for t in target:
        sent.append(token_to_char[int(t)])
    print('Spoken sentence:', ''.join(sent))

    word = []
    for t in range(output.shape[0]):
        chars = output[t,0,:]
        cur_char = token_to_char[torch.argmax(chars).item()]
        if cur_char != "_":
            word.append(cur_char)
    print('Outptut:', ''.join(word))