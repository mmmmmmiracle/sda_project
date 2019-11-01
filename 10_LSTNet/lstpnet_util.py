import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms, utils
from sklearn.preprocessing import MinMaxScaler

def get_train_val_test_idxs(args):
    data_len = len(args.data) - args.window_size - 1
    train_size, val_size, test_size = 0.6, 0.2, 0.2
    train_val_idxs = np.array( range( 0, int( data_len * (1- test_size) ) ) )
    np.random.shuffle( train_val_idxs )
    idxs = {}
    idxs['train']= train_val_idxs[:int(len(train_val_idxs) * train_size / (1-test_size))]
    idxs['val'] = train_val_idxs[int(len(train_val_idxs) * train_size / (1-test_size)):]
#     print(train_idxs.shape,val_idxs.shape)
    idxs['test'] = np.array(range(int( data_len * (1-test_size)),data_len))
    return idxs

class Args():
    data_file_name = 'data/reservoir.npz'
    data = np.load(data_file_name)['a']
    
    scaler = 'minmax'
    
    feature_dim = 5
    window_size = 24 * 7
    
    cnn_kernal_size = 6
    cnn_out_channel = 100
    
    gru1_hide_dim = 100
    gru1_layers = 1
    gru2_hide_dim = 10
    gru2_layers = 1
    
    formerN_time_stamp = 24
    period_cycle = 24
    
    dropout = 0.2
    output_fun = 'tanh'
    
    use_gpu = torch.cuda.is_available()
    batch_size = 1024
    
    metrics = [nn.L1Loss(reduction='sum') ,nn.MSELoss(reduce='sum')]

args = Args()
    
class LSTPNet(nn.Module):
    def __init__(self,args):
        super(LSTPNet,self).__init__()
        self.feature_dim = args.feature_dim
        self.window_size = args.window_size
        self.cnn_out_channel = args.cnn_out_channel
        self.cnn_kernal_size = args.cnn_kernal_size
        self.gru1_hide_dim = args.gru1_hide_dim
        self.gru1_layers = 1
        self.gru2_hide_dim = args.gru2_hide_dim
        self.gru2_layers = 1
        self.formerN_time_stamp = args.formerN_time_stamp
        self.period_cycle = args.period_cycle
        self.num_period_cycle = int((self.window_size - self.cnn_kernal_size + 1)/self.period_cycle)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p = args.dropout)
        
        self.conv1d = nn.Conv1d(self.feature_dim,self.cnn_out_channel,kernel_size=self.cnn_kernal_size)
        self.gru1 = nn.GRU(self.cnn_out_channel,self.gru1_hide_dim,self.gru1_layers)
        if(self.period_cycle > 0):
            self.gru2 = nn.GRU(self.cnn_out_channel,self.gru2_hide_dim,self.gru2_layers)
            self.linear_out = nn.Linear(self.gru1_hide_dim + self.period_cycle * self.gru2_hide_dim, self.feature_dim)
        else:
            self.linear_out = nn.Linear(self.gru1_hide_dim, self.feature_dim)
        if(self.formerN_time_stamp > 0):
            self.linear_ar = nn.Linear(self.period_cycle,1)
        self.output = None
        if(args.output_fun == 'sigmoid'):
            self.output = nn.Sigmoid()
        elif(args.output_fun == 'relu'):
            self.output = nn.ReLU()
        else:
            pass
        
    def forward(self,x):
#         print(x)
        batch_size = x.size(0)
        x = x.permute(0,2,1).contiguous()
#         print(x.shape)

        #cnn
        c = self.relu(self.conv1d(x))
#         c = self.tanh(self.conv1d(x))
#         print(c.shape)

        #rnn
        r = c.permute(2,0,1).contiguous()
#         print(c.shape)
        _,r = self.gru1(r)
        r = self.tanh(r)
        r = self.dropout(torch.squeeze(r,0))

        #skip-rnn
        if self.period_cycle > 0:
#             print(c.shape)
            s = c[:,:,int(-self.num_period_cycle * self.period_cycle):].contiguous()
#             print(s.shape)
            s = s.view(batch_size,self.cnn_out_channel,self.num_period_cycle,self.period_cycle)
#             print(s.shape)
            s = s.permute(2,0,3,1).contiguous()
#             print(s.shape)
            s = s.view(self.num_period_cycle,self.period_cycle * batch_size,self.cnn_out_channel)
            _,s = self.gru2(s)
            s = s.view(batch_size,self.period_cycle * self.gru2_hide_dim)
            s = self.tanh(s)
            s = self.dropout(s)
            r = torch.cat((r,s),dim=1)

        res = self.linear_out(r)

        #ar
        if self.formerN_time_stamp > 0:
            x = x.permute(0,2,1).contiguous()
#             print(x.shape)
            z = x[:,-self.formerN_time_stamp:,:]
#             print(z.shape)
            z = z.permute(0,2,1).contiguous().view(-1,self.formerN_time_stamp)
#             print(z.shape)
            z = self.linear_ar(z)
#             print(z.shape)
            z = z.view(batch_size,self.feature_dim)
            z = self.tanh(z)
#             print(res.shape,z.shape)
            res = torch.add(res , z)

        if self.output:
            self.output(res)
        return res

class LstpDataset(Dataset):
    def __init__(self, args, transforms, indexs):
        self.data = args.data
        self.indexs = indexs
        self.transforms = transforms
#         print(self.__len__())
        
    def __len__(self):
        return len(self.indexs)
    
    def __getitem__(self,idx):
        idx = self.indexs[idx]
#         print(idx)
        x = self.data[idx:idx+args.window_size,:]
        y = self.data[idx+args.window_size,:].reshape(1,-1)
        sample = {'x':x,'y':y}
        if self.transforms:
            sample = self.transforms(sample)
        return sample
    
class Normalize(object):
    def __init__(self,args):
        if args.scaler is 'minmax':
            self.scaler = MinMaxScaler()
    
    def __call__(self,sample):
        x = sample['x']
        y = sample['y']
        x = self.scaler.fit_transform(x)
        y = self.scaler.transform(y)
        sample['x'] = x
        sample['y'] = y
        return sample

class ToTensor(object):
    def __call__(self,sample):
        x = sample['x']
        y = sample['y']
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        y = y.view(-1)
        sample['x'] = x
        sample['y'] = y
        return sample

class MyLoss(nn.Module):
    def __init__(self,l1_weight,l2_weight,reduction='mean'):
        super(MyLoss,self).__init__()
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.reduction = reduction
        
    def forward(self,yhat,y):
        if self.reduction=='mean':
            l1_part = self.l1_weight * torch.mean(torch.abs(yhat-y))
            l2_part = self.l2_weight * torch.mean(torch.mul(yhat,y))
        elif self.reduction=='sum':
            l1_part = self.l1_weight * torch.sum(torch.abs(yhat-y))
            l2_part = self.l2_weight * torch.sum(torch.mul(yhat,y))
        return l1_part+l2_part
    
    

# from ignite.engine import create_supervised_evaluator,create_supervised_trainer
# from ignite.metrics import MeanSquaredError
# from ignite.engine import Events
# from ignite.handlers import ModelCheckpoint
# def train_engine():
#     lstpnet = LSTPNet(args)
#     optimizer = torch.optim.Adam(lstpnet.parameters()) 
#     criterion = MyLoss(l1_weight=2,l2_weight=0.02,reduction='mean')
#     max_epochs = 1000
#     validate_every = 10
#     checkpoint_every = 10
    
#     trainer = create_supervised_trainer(lstpnet,optimizer,criterion,device='cuda')
#     evaluator = create_supervised_evaluator(lstpnet,metrics={'mse':MeanSquaredError()},device='cuda')
    
#     @trainer.on(Events.ITERATION_COMPLETED)
#     def validate(trainer):
#         if trainer.state.iteration % validate_every == 0:
#             evaluator.run(val_dataloader)
#             metrics = evaluator.run(val_dataloader).metrics['mse']
#             print(f"iteration: {trainer.state.iteration}, mse: {metrics['mse']}")
#     checkpointer = ModelCheckpoint('checkpoints/','lstp_er',save_interval=checkpoint_every,create_dir=True)
#     trainer.add_event_handler(Events.ITERATION_COMPLETED,checkpointer,{'lstp_er':lstpnet})
#     trainer.run(train_dataloader,max_epochs=max_epochs)

def mape(y_true, y_pred):
    """
    参数:
    y_true -- 测试集目标真实值
    y_pred -- 测试集目标预测值
    
    返回:
    mape -- MAPE 评价指标
    """
    
    n = len(y_true)
    mape = sum(np.abs((y_true - y_pred)/y_true))/n*100
    return mape