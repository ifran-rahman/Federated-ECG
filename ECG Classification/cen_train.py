import torch
from torch.utils.data import DataLoader
from torch import nn,optim
import sys
from tqdm import tqdm
import pandas as pd
import numpy as np

# define a Dataloader function
def my_DataLoader(train_root, test_root, batch_size = 100, val_split_factor = 0.2):

    train_df = pd.read_csv(train_root, header=None)
    test_df = pd.read_csv(test_root, header=None)

    train_data = train_df.to_numpy()
    test_data = test_df.to_numpy()

    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_data[:, :-1]).float(),
                                                   torch.from_numpy(train_data[:, -1]).long(),)
    test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(test_data[:, :-1]).float(),
                                                  torch.from_numpy(test_data[:, -1]).long())

    train_len = train_data.shape[0]
    val_len = int(train_len * val_split_factor)
    train_len -= val_len

    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_len, val_len])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    num_examples =  {'trainset': train_len, 
                    'testset': val_len}


    return train_loader, val_loader, test_loader, num_examples


#define the ecg_net model
class  ecg_net(nn.Module):

    def __init__(self, num_of_class):
        super(ecg_net, self).__init__()

        self.model = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool1d(2),

        )

        self.linear = nn.Sequential(
            nn.Linear(2944,500),
            nn.LeakyReLU(inplace=True),
            nn.Linear(500, num_of_class),

        )


    def forward(self,x):
        x = x.unsqueeze(1)
        x = self.model(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        #x [b, 2944]
        # print(x.shape)
        x = self.linear(x)

        return x


# hyperparameters
batch_size=1000
lr = 3e-3
epochs = 10
val_split_factor = 0.2
torch.manual_seed(1234)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))

def evalute(model, loader):
    model.eval()

    correct = 0
    total = len(loader.dataset)
    val_bar = tqdm(loader, file=sys.stdout)
    for x, y in val_bar:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
        correct += torch.eq(pred, y).sum().float().item()

    return correct / total

def train_client(model, train_loader, valid_loader, epochs=1):

    # model = ecg_net(2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()

    best_acc, best_epoch = 0, 0
    global_step = 0

    
    for epoch in range(epochs):
        
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, (x, y) in enumerate(train_bar):
            print("first batch entered")
            # x: [b, 187], y: [b]
            x, y = x.to(device), y.to(device)

            model.train()
            
            logits = model(x)
            loss = criteon(logits, y)
            
            optimizer.zero_grad()
            loss.backward()
           
            # for param in model.parameters():
            #     print(param.grad)

            optimizer.step()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

            global_step += 1

        if epoch % 1 == 0:  # You can change the validation frequency as you wish

            val_acc = evalute(model, valid_loader)
            
            print('val_acc = ',val_acc)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc

                # torch.save(model.state_dict(), 'best_client_model.mdl')

        print("Global steps", global_step)

    print('best acc:', best_acc, 'best epoch:', best_epoch)

    model.load_state_dict(torch.load('best.mdl'))
    print('loaded from ckpt!')

def validate(model, testloader, criterion):
    return 0,0


def prepare__dataset(abnormal, normal, val_split_factor):

    abnormal = abnormal.drop([187], axis=1)
    normal = normal.drop([187], axis=1)

    y_abnormal = np.ones((abnormal.shape[0]))
    y_abnormal = pd.DataFrame(y_abnormal)

    y_normal = np.zeros((normal.shape[0]))
    y_normal = pd.DataFrame(y_normal)

    x = pd.concat([abnormal, normal], sort=True)
    y = pd.concat([y_abnormal, y_normal] ,sort=True)

    x = x.to_numpy()
    y = y[0].to_numpy()

    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x).float(),
                                                torch.from_numpy(y).long())

    train_len = x.shape[0]
    val_len = int(train_len * val_split_factor)
    train_len -= val_len

    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_len, val_len])

    num_examples =  {'trainset': train_len, 
                    'testset': val_len}

    return train_dataset, val_dataset, num_examples


def main():
    
    # load dataset
    abnormal = pd.read_csv('datasets/ptbdb_abnormal.csv', header = None)
    normal = pd.read_csv('datasets/ptbdb_normal.csv', header = None)

    train_dataset, val_dataset, _ = prepare__dataset(abnormal=abnormal, normal=normal, val_split_factor=val_split_factor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    model = ecg_net(2).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()

    best_acc, best_epoch = 0, 0
    global_step = 0

    for epoch in range(epochs):
    
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, (x, y) in enumerate(train_bar):
            # x: [b, 187], y: [b]
            x, y = x.to(device), y.to(device)

            model.train()
            logits = model(x)
            loss = criteon(logits, y)

            optimizer.zero_grad()
            loss.backward()

            # for param in model.parameters():
            #     print(param.grad)

            optimizer.step()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)
            global_step += 1

        if epoch % 1 == 0:  # You can change the validation frequency as you wish

            val_acc = evalute(model, val_loader)
            
            print('val_acc = ',val_acc)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc

                torch.save(model.state_dict(), 'best.mdl')


    print('best acc:', best_acc, 'best epoch:', best_epoch)

    model.load_state_dict(torch.load('best.mdl'))
    print('loaded from ckpt!')

    # test_acc = evalute(model, test_loader)
    # print('test acc:', test_acc)


if __name__ == '__main__':
    main()