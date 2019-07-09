import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import pandas as pd
import math
import time
from datetime import datetime
import random
import sys
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import *
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch_scatter import *
from torch_geometric.utils import scatter_


from dscribe.descriptors import ACSF
from dscribe.core.system import System

from timeit import default_timer as timer


COMP_PATH = '/home/alexpartisan/Work/Data/Competition/CHAMP'
DATA_DIR = COMP_PATH+'/input/champs-scalar-coupling'
RES_DIR = COMP_PATH+'/GNN/results'
RESULT_DIR = COMP_PATH+'/result'


PROJECT_PATH = os.path.dirname(os.path.realpath(__file__).replace('/lib',''))

IDENTIFIER = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')



COUPLING_TYPE_STATS=[
    #type   #mean, std, min, max
    '1JHC',  94.9761528641869,   18.27722399839607,   66.6008,   204.8800,
    '2JHC',  -0.2706244378832,    4.52360876732858,  -36.2186,    42.8192,
    '3JHC',   3.6884695895355,    3.07090647005439,  -18.5821,    76.0437,
    '1JHN',  47.4798844844683,   10.92204561670947,   24.3222,    80.4187,
    '2JHN',   3.1247536134185,    3.67345877025737,   -2.6209,    17.7436,
    '3JHN',   0.9907298624944,    1.31538940138001,   -3.1724,    10.9712,
    '2JHH', -10.2866051639817,    3.97960190019757,  -35.1761,    11.8542,
    '3JHH',   4.7710233597359,    3.70498129755812,   -3.0205,    17.4841,
]
NUM_COUPLING_TYPE = len(COUPLING_TYPE_STATS)//5

COUPLING_TYPE_MEAN = [ COUPLING_TYPE_STATS[i*5+1] for i in range(NUM_COUPLING_TYPE)]
COUPLING_TYPE_STD  = [ COUPLING_TYPE_STATS[i*5+2] for i in range(NUM_COUPLING_TYPE)]
COUPLING_TYPE      = [ COUPLING_TYPE_STATS[i*5  ] for i in range(NUM_COUPLING_TYPE)]

SYMBOL=['H', 'C', 'N', 'O', 'F']

ACSF_GENERATOR = ACSF(
    species = SYMBOL,
    rcut = 6.0,
    g2_params=[[1, 1], [1, 2], [1, 3]],
    g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]],
)

EDGE_DIM   =  6
NODE_DIM   = 93 ##  93  13
NUM_TARGET =  8


# EDGE_DIM   =  80
# NODE_DIM   = 16 ##  93  13
# NUM_TARGET =  8


#---------------------------------------------------------------------------------
COMMON_STRING ='@%s:  \n' % os.path.basename(__file__)

if 1:
    SEED = int(time.time()) #35202   #35202  #123  #
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    COMMON_STRING += '\tset random seed\n'
    COMMON_STRING += '\t\tSEED = %d\n'%SEED

    torch.backends.cudnn.benchmark     = True  ##uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms. -
    torch.backends.cudnn.enabled       = True
    torch.backends.cudnn.deterministic = True

    COMMON_STRING += '\tset cuda environment\n'
    COMMON_STRING += '\t\ttorch.__version__              = %s\n'%torch.__version__
    COMMON_STRING += '\t\ttorch.version.cuda             = %s\n'%torch.version.cuda
    COMMON_STRING += '\t\ttorch.backends.cudnn.version() = %s\n'%torch.backends.cudnn.version()
    try:
        COMMON_STRING += '\t\tos[\'CUDA_VISIBLE_DEVICES\']     = %s\n'%os.environ['CUDA_VISIBLE_DEVICES']
        NUM_CUDA_DEVICES = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    except Exception:
        COMMON_STRING += '\t\tos[\'CUDA_VISIBLE_DEVICES\']     = None\n'
        NUM_CUDA_DEVICES = 1

    COMMON_STRING += '\t\ttorch.cuda.device_count()      = %d\n'%torch.cuda.device_count()
    #print ('\t\ttorch.cuda.current_device()    =', torch.cuda.current_device())


COMMON_STRING += '\n'


def time_to_str(t, mode='min'):
    if mode=='min':
        t  = int(t)/60
        hr = t//60
        min = t%60
        return '%2d hr %02d min'%(hr,min)

    elif mode=='sec':
        t   = int(t)
        min = t//60
        sec = t%60
        return '%2d min %02d sec'%(min,sec)


    else:
        raise NotImplementedError


# http://stackoverflow.com/questions/34950201/pycharm-print-end-r-statement-not-working
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout  #stdout
        self.file = None

    def open(self, file, mode=None):
        if mode is None: mode ='w'
        self.file = open(file, mode)

    def write(self, message, is_terminal=1, is_file=1 ):
        if '\r' in message: is_file=0

        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()
            #time.sleep(1)

        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


def read_pickle_from_file(pickle_file):
    with open(pickle_file,'rb') as f:
        x = pickle.load(f)
    return x

def write_pickle_to_file(pickle_file, x):
    with open(pickle_file, 'wb') as f:
        pickle.dump(x, f, pickle.HIGHEST_PROTOCOL)

######################################################################################
############################################################################# Model
######################################################################################

class LinearBn(nn.Module):
    def __init__(self, in_channel, out_channel, act=None):
        super(LinearBn, self).__init__()
        self.linear = nn.Linear(in_channel, out_channel, bias=False)
        self.bn   = nn.BatchNorm1d(out_channel,eps=1e-05, momentum=0.1)
        self.act  = act

    def forward(self, x):
        x = self.linear(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x

class GraphConv(nn.Module):
    def __init__(self, node_dim, edge_dim ):
        super(GraphConv, self).__init__()

        # edge_dim -> node_dim * node_dim
        self.encoder = nn.Sequential(
            LinearBn(edge_dim, 256),
            nn.ReLU(inplace=True),
            LinearBn(256, 256),
            nn.ReLU(inplace=True),
            LinearBn(256, 128),
            nn.ReLU(inplace=True),
            LinearBn(128, node_dim * node_dim),
            #nn.ReLU(inplace=True),
        )

        self.gru  = nn.GRU(node_dim, node_dim, batch_first=False, bidirectional=False)
        self.bias = nn.Parameter(torch.Tensor(node_dim))
        self.bias.data.uniform_(-1.0 / math.sqrt(node_dim), 1.0 / math.sqrt(node_dim))


    def forward(self, node, edge_index, edge, hidden):
        num_node, node_dim = node.shape
        num_edge, edge_dim = edge.shape
        edge_index = edge_index.t().contiguous()

        #1. message :  m_j = SUM_i f(n_i, n_j, e_ij)  where i is neighbour(j)
        x_i     = torch.index_select(node, 0, edge_index[0])
        edge    = self.encoder(edge).view(-1, node_dim, node_dim)
        #message = x_i.view(-1,node_dim,1)*edge
        #message = message.sum(1)


       # if tensor1 is a (j×1×n×m) tensor and tensor2 is a (k×m×p) tensor
       # out will be a (j×k×n×p) tensor.
        message = x_i.view(-1,1,node_dim)@edge  # @ operator as matrix-vector and matrix-matrix multiplication
        message = message.view(-1,node_dim)
        message = scatter_('mean', message, edge_index[1], dim_size=num_node)
        message = F.relu(message +self.bias)

        #2. update: n_j = f(n_j, m_j)
        update = message

        #batch_first=True
        update, hidden = self.gru(update.view(1,-1,node_dim), hidden)
        update = update.view(-1,node_dim)

        return update, hidden

class Set2Set(torch.nn.Module):

    def softmax(self, x, index, num=None):
        x = x -  scatter_max(x, index, dim=0, dim_size=num)[0][index]
        x = x.exp()
        x = x / (scatter_add(x, index, dim=0, dim_size=num)[index] + 1e-16)
        return x

    def __init__(self, in_channel, processing_step=1):
        super(Set2Set, self).__init__()
        num_layer = 1
        out_channel = 2 * in_channel

        self.processing_step = processing_step
        self.in_channel  = in_channel
        self.out_channel = out_channel
        self.num_layer   = num_layer
        self.lstm = torch.nn.LSTM(out_channel, in_channel, num_layer)
        self.lstm.reset_parameters()

    def forward(self, x, batch_index):
        batch_size = batch_index.max().item() + 1

        h = (x.new_zeros((self.num_layer, batch_size, self.in_channel)),
             x.new_zeros((self.num_layer, batch_size, self.in_channel)))

        q_star = x.new_zeros(batch_size, self.out_channel)
        for i in range(self.processing_step):
            q, h = self.lstm(q_star.unsqueeze(0), h)
            q = q.view(batch_size, -1)

            e = (x * q[batch_index]).sum(dim=-1, keepdim=True) #shape = num_node x 1
            a = self.softmax(e, batch_index, num=batch_size)   #shape = num_node x 1
            r = scatter_add(a * x, batch_index, dim=0, dim_size=batch_size) #apply attention #shape = batch_size x ...
            q_star = torch.cat([q, r], dim=-1)

        return q_star





# EDGE_DIM   = 80
# NODE_DIM   = 16 ##  93  13
# NUM_TARGET =  8

#message passing
class Net(torch.nn.Module):
    def __init__(self, node_dim=13, edge_dim=5, num_target=8):
    # def __init__(self, node_dim=NODE_DIM, edge_dim=EDGE_DIM, num_target=NUM_TARGET):
        super(Net, self).__init__()
        self.num_propagate = 6
        self.num_s2s = 6


        # node_dim -> 128
        self.preprocess = nn.Sequential(
            LinearBn(node_dim, 128),
            nn.ReLU(inplace=True),
            LinearBn(128, 128),
            nn.ReLU(inplace=True),
        )

        self.propagate = GraphConv(128, edge_dim)
        self.set2set = Set2Set(128, processing_step=self.num_s2s)


        #predict coupling constant
        self.predict = nn.Sequential(
            LinearBn(4*128, 1024),  #node_hidden_dim
            nn.ReLU(inplace=True),
            LinearBn(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_target),
        )

    def forward(self, node, edge, edge_index, node_index, coupling_index):

        num_node, node_dim = node.shape
        num_edge, edge_dim = edge.shape

        node   = self.preprocess(node)
        hidden = node.view(1,num_node,-1)

        for i in range(self.num_propagate):
            node, hidden =  self.propagate(node, edge_index, edge, hidden)

        pool = self.set2set(node, node_index)

        #---
        num_coupling = len(coupling_index)
        coupling_atom0_index, coupling_atom1_index, coupling_type_index, coupling_batch_index = \
            torch.split(coupling_index,1,dim=1)

        pool  = torch.index_select( pool, dim=0, index=coupling_batch_index.view(-1))
        node0 = torch.index_select( node, dim=0, index=coupling_atom0_index.view(-1))
        node1 = torch.index_select( node, dim=0, index=coupling_atom1_index.view(-1))

        predict = self.predict(torch.cat([pool,node0,node1],-1))
        predict = torch.gather(predict, 1, coupling_type_index).view(-1)
        return predict



# def criterion(predict, coupling_value):
#     predict = predict.view(-1)
#     coupling_value = coupling_value.view(-1)
#     assert(predict.shape==coupling_value.shape)
#
#     loss = F.mse_loss(predict, coupling_value)
#     return loss

def criterion(predict, truth):
    predict = predict.view(-1)
    truth   = truth.view(-1)
    assert(predict.shape==truth.shape)

    loss = torch.abs(predict-truth)
    loss = loss.mean()
    loss = torch.log(loss)
    return loss


##################################################################################################################

def make_dummy_data(node_dim, edge_dim, num_target, batch_size):

    #dummy data
    num_node = []
    num_edge = []

    node = []
    edge = []
    edge_index = []
    node_index = []

    coupling_value = []
    coupling_atom_index  = []
    coupling_type_index  = []
    coupling_batch_index = []


    for b in range(batch_size):
        node_offset = sum(num_node)
        edge_offset = sum(num_edge)

        N = np.random.choice(10)+8
        E = np.random.choice(10)+16
        node.append(np.random.uniform(-1,1,(N,node_dim)))
        edge.append(np.random.uniform(-1,1,(E,edge_dim)))

        edge_index.append(np.random.choice(N, (E,2))+node_offset)
        node_index.append(np.array([b]*N))

        #---
        C = np.random.choice(10)+1
        coupling_value.append(np.random.uniform(-1,1, C))
        coupling_atom_index.append(np.random.choice(N,(C,2))+node_offset)
        coupling_type_index.append(np.random.choice(num_target, C))
        coupling_batch_index.append(np.array([b]*C))

        #---
        num_node.append(N)
        num_edge.append(E)


    node = torch.from_numpy(np.concatenate(node)).float().cuda()
    edge = torch.from_numpy(np.concatenate(edge)).float().cuda()
    edge_index = torch.from_numpy(np.concatenate(edge_index)).long().cuda()
    node_index = torch.from_numpy(np.concatenate(node_index)).long().cuda()

    #---
    coupling_value = torch.from_numpy(np.concatenate(coupling_value)).float().cuda()
    coupling_index = np.concatenate([
        np.concatenate(coupling_atom_index),
        np.concatenate(coupling_type_index).reshape(-1,1),
        np.concatenate(coupling_batch_index).reshape(-1,1),
    ],-1)
    coupling_index = torch.from_numpy(np.array(coupling_index)).long().cuda()


    return node, edge, edge_index, node_index, coupling_value, coupling_index



def run_check_net():

    #dummy data
    node_dim = 5
    edge_dim = 7
    num_target = 8
    batch_size = 16
    node, edge, edge_index, node_index, coupling_value, coupling_index = \
        make_dummy_data(node_dim, edge_dim, num_target, batch_size)

    print('batch_size ', batch_size)
    print('----')
    print('node',node.shape)
    print('edge',edge.shape)
    print('edge_index',edge_index.shape)
    print('node_index',node_index.shape)
    print('----')

    print('coupling_index',coupling_index.shape)
    print('')

    #---
    net = Net(node_dim=node_dim, edge_dim=edge_dim, num_target=num_target).cuda()
    net = net.eval()



    predict = net(node, edge, edge_index, node_index, coupling_index)

    print('predict: ', predict.shape)
    print(predict)
    print('')

    if 0:
        keys = list(net.state_dict().keys())
        sorted(keys)
        for k in keys:
            if '.num_batches_tracked' in k:
                continue
            print(' \'%s\','%k)



def run_check_train():

    node_dim = 15
    edge_dim =  5
    num_target = 12
    batch_size = 64
    node, edge, edge_index, node_index, coupling_value, coupling_index = \
        make_dummy_data(node_dim, edge_dim, num_target, batch_size)


    net = Net(node_dim=node_dim, edge_dim=edge_dim, num_target=num_target).cuda()
    net = net.eval()


    predict = net(node, edge, edge_index, node_index, coupling_index)
    loss = criterion(predict, coupling_value)


    print('*loss = %0.5f'%( loss.item(),))
    print('')

    print('predict: ', predict.shape)
    print(predict)
    print(coupling_value)
    print('')

    # dummy sgd to see if it can converge ...
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                      lr=0.01, momentum=0.9, weight_decay=0.0001)

    print('--------------------')
    print('[iter ]  loss       ')
    print('--------------------')

    i=0
    optimizer.zero_grad()
    while i<=500:
        net.train()
        optimizer.zero_grad()

        predict = net(node, edge, edge_index, node_index, coupling_index)
        loss = criterion(predict, coupling_value)

        loss.backward()
        optimizer.step()

        if i%10==0:
            print('[%05d] %8.5f  '%(
                i,
                loss.item(),
            ))
        i = i+1
    print('')

    #check results
    print(predict[:5])
    print(coupling_value[:5])
    print('')

######################################################################################
############################################################################# Train
######################################################################################
def compute_kaggle_metric( predict, coupling_value, coupling_type):

    mae     = [None]*NUM_COUPLING_TYPE
    log_mae = [None]*NUM_COUPLING_TYPE
    diff = np.fabs(predict-coupling_value)
    for t in range(NUM_COUPLING_TYPE):
        index = np.where(coupling_type==t)[0]
        if len(index)>0:
            m = diff[index].mean()
            log_m = np.log(m+1e-8)

            mae[t] = m
            log_mae[t] = log_m
        else:
            pass

    return mae, log_mae

def do_valid(net, valid_loader):

    valid_num = 0
    valid_predict = []
    valid_coupling_type  = []
    valid_coupling_value = []

    valid_loss = 0
    for b, (node, edge, edge_index, node_index, coupling_value, coupling_index, infor) in enumerate(valid_loader):

        #if b==5: break
        net.eval()
        node = node.cuda()
        edge = edge.cuda()
        edge_index = edge_index.cuda()
        node_index = node_index.cuda()

        coupling_value = coupling_value.cuda()
        coupling_index = coupling_index.cuda()

        with torch.no_grad():
            predict = net(node, edge, edge_index, node_index, coupling_index)
            loss = criterion(predict, coupling_value)

        #---
        batch_size = len(infor)
        valid_predict.append(predict.data.cpu().numpy())
        valid_coupling_type.append(coupling_index[:,2].data.cpu().numpy())
        valid_coupling_value.append(coupling_value.data.cpu().numpy())

        valid_loss += batch_size*loss.item()
        valid_num  += batch_size

        print('\r %8d /%8d'%(valid_num, len(valid_loader.dataset)),end='',flush=True)

        pass  #-- end of one data loader --
    assert(valid_num == len(valid_loader.dataset))
    #print('')
    valid_loss = valid_loss/valid_num

    #compute
    predict = np.concatenate(valid_predict)
    coupling_value = np.concatenate(valid_coupling_value)
    coupling_type  = np.concatenate(valid_coupling_type).astype(np.int32)
    mae, log_mae   = compute_kaggle_metric( predict, coupling_value, coupling_type,)

    num_target = NUM_COUPLING_TYPE
    for t in range(NUM_COUPLING_TYPE):
        if mae[t] is None:
            mae[t] = 0
            log_mae[t]  = 0
            num_target -= 1

    mae_mean, log_mae_mean = sum(mae)/num_target, sum(log_mae)/num_target
    #list(np.stack([mae, log_mae]).T.reshape(-1))

    valid_loss = log_mae + [valid_loss,mae_mean, log_mae_mean, ]
    return valid_loss



class ChampsDataset(Dataset):
    def __init__(self, split, csv, mode, augment=None):

        self.split   = split
        self.csv     = csv
        self.mode    = mode
        self.augment = augment

        self.df = pd.read_csv(DATA_DIR + '/%s.csv'%csv)

        if split is not None:
            self.id = np.load(DATA_DIR + '/split/%s'%split,allow_pickle=True)
        else:
            self.id = self.df.molecule_name.unique()

        #zz=0
        #self.dummy_graph = read_pickle_from_file(DATA_DIR + '/structure/graph/dsgdb9nsd_000001.pickle')

    def __str__(self):
            string = ''\
            + '\tmode   = %s\n'%self.mode \
            + '\tsplit  = %s\n'%self.split \
            + '\tcsv    = %s\n'%self.csv \
            + '\tlen    = %d\n'%len(self)

            return string

    def __len__(self):
        return len(self.id)


    def __getitem__(self, index):

        molecule_name = self.id[index]
        graph_file = '/home/alexpartisan/Work/Data/Competition/CHAMP/input/graph/%s.pickle'%molecule_name
        graph = read_pickle_from_file(graph_file)
        assert(graph.molecule_name==molecule_name)

        # ##filter only J link
        # if 0:
        #     # 1JHC,     2JHC,     3JHC,     1JHN,     2JHN,     3JHN,     2JHH,     3JHH
        #     mask = np.zeros(len(graph.coupling.type),np.bool)
        #     for t in ['1JHC',     '2JHH']:
        #         mask += (graph.coupling.type == COUPLING_TYPE.index(t))
        #
        #     graph.coupling.id = graph.coupling.id [mask]
        #     graph.coupling.contribution = graph.coupling.contribution [mask]
        #     graph.coupling.index = graph.coupling.index [mask]
        #     graph.coupling.type = graph.coupling.type [mask]
        #     graph.coupling.value = graph.coupling.value [mask]

        if 1:
            atom = System(symbols =graph.axyz[0], positions=graph.axyz[1])
            acsf = ACSF_GENERATOR.create(atom)
            graph.node += [acsf,]


        # if 1:
        #     graph.edge = graph.edge[:-1]

        graph.node = np.concatenate(graph.node,-1)
        graph.edge = np.concatenate(graph.edge,-1)
        return graph



class NullScheduler():
    def __init__(self, lr=0.01):
        super(NullScheduler, self).__init__()
        self.lr    = lr
        self.cycle = 0

    def __call__(self, time):
        return self.lr

    def __str__(self):
        string = 'NullScheduler\n' \
                + 'lr=%0.5f '%(self.lr)
        return string


# net ------------------------------------
# https://github.com/pytorch/examples/blob/master/imagenet/main.py ###############
def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
       lr +=[ param_group['lr'] ]

    assert(len(lr)==1) #we support only one param_group
    lr = lr[0]

    return lr

def null_collate(batch):

    batch_size = len(batch)

    node = []
    edge = []
    edge_index = []
    node_index = []

    coupling_value = []
    coupling_atom_index  = []
    coupling_type_index  = []
    coupling_batch_index = []
    infor = []

    offset = 0
    for b in range(batch_size):
        graph = batch[b]
        #print(graph.molecule_name)

        num_node = len(graph.node)
        node.append(graph.node)
        edge.append(graph.edge)
        edge_index.append(graph.edge_index+offset)
        node_index.append(np.array([b]*num_node))

        num_coupling = len(graph.coupling.value)
        coupling_value.append(graph.coupling.value)
        coupling_atom_index.append(graph.coupling.index+offset)
        coupling_type_index.append (graph.coupling.type)
        coupling_batch_index.append(np.array([b]*num_coupling))

        infor.append((graph.molecule_name, graph.smiles, graph.coupling.id))
        offset += num_node
        #print(num_node, len(coupling_batch_index))

    node = torch.from_numpy(np.concatenate(node)).float()
    edge = torch.from_numpy(np.concatenate(edge)).float()
    edge_index = torch.from_numpy(np.concatenate(edge_index).astype(np.int32)).long()
    node_index = torch.from_numpy(np.concatenate(node_index)).long()

    coupling_value = torch.from_numpy(np.concatenate(coupling_value)).float()
    coupling_index = np.concatenate([
        np.concatenate(coupling_atom_index),
        np.concatenate(coupling_type_index).reshape(-1,1),
        np.concatenate(coupling_batch_index).reshape(-1,1),
    ],-1)
    coupling_index = torch.from_numpy(coupling_index).long()
    return node, edge, edge_index, node_index, coupling_value, coupling_index, infor


def run_train(out_dir, initial_checkpoint=None):


    # Adjust learning rate
    # schduler = NullScheduler(lr=0.001)
    schduler = NullScheduler(lr=0.0001)

    ## setup  -----------------------------------------------------------------------------
    os.makedirs(out_dir +'/checkpoint', exist_ok=True)
    os.makedirs(out_dir +'/train', exist_ok=True)
    os.makedirs(out_dir +'/backup', exist_ok=True)
    # backup_project_as_zip(PROJECT_PATH, out_dir +'/backup/code.train.%s.zip'%IDENTIFIER)

    log = Logger()
    log.open(out_dir+'/log.train.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('\t%s\n' % COMMON_STRING)
    log.write('\n')

    log.write('\tSEED         = %u\n' % SEED)
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    log.write('\t__file__     = %s\n' % __file__)
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')


    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')
    # batch_size = 20 #*2 #280*2 #256*4 #128 #256 #512  #16 #32
    batch_size = 16

    train_dataset = ChampsDataset(
                csv='train',
                mode ='train',
                # split='debug_split_by_mol.1000.npy', #
                split='train_split_by_mol_0_56668.npy',
                augment=None,
    )
    train_loader  = DataLoader(
                train_dataset,
                #sampler     = SequentialSampler(train_dataset),
                sampler     = RandomSampler(train_dataset),
                batch_size  = batch_size,
                drop_last   = True,
                num_workers = 16,
                pin_memory  = True,
                collate_fn  = null_collate
    )

    valid_dataset = ChampsDataset(
                csv='train',
                mode='train',
                # split='debug_split_by_mol.1000.npy', # #,None
                split='valid_split_by_mol_0_28335.npy',
                augment=None,
    )
    valid_loader = DataLoader(
                valid_dataset,
                #sampler     = SequentialSampler(valid_dataset),
                sampler     = RandomSampler(valid_dataset),
                batch_size  = batch_size,
                drop_last   = False,
                num_workers = 0,
                pin_memory  = True,
                collate_fn  = null_collate
    )


    assert(len(train_dataset)>=batch_size)
    log.write('batch_size = %d\n'%(batch_size))
    log.write('train_dataset : \n%s\n'%(train_dataset))
    log.write('valid_dataset : \n%s\n'%(valid_dataset))
    log.write('\n')

    ## net ----------------------------------------
    log.write('** net setting **\n')
    net = Net(node_dim=NODE_DIM,edge_dim=EDGE_DIM, num_target=NUM_TARGET).cuda()

    log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
    if initial_checkpoint is not None:
        # Load all tensors onto GPU 1
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))

    log.write('%s\n'%(type(net)))
    log.write('\n')

    # pretrain_file = '/root/share/project/kaggle/2019/champs_scalar/result/backup/00370000_model.pth'
    # load_pretrain(net,pretrain_file)

    ## optimiser ----------------------------------
    # if 0: ##freeze
    #     for p in net.encoder1.parameters(): p.requires_grad = False
    #     pass

    #net.set_mode('train',is_freeze_bn=True)
    #-----------------------------------------------

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),lr=schduler(0))
    #optimizer = torch.optim.RMSprop(net.parameters(), lr =0.0005, alpha = 0.95)
    #optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=schduler(0), momentum=0.9, weight_decay=0.0001)

    iter_accum  = 1
    num_iters   = 3000  *1000
    iter_smooth = 50
    iter_log    = 500
    iter_valid  = 500
    iter_save   = [0, num_iters-1]\
                   + list(range(0, num_iters, 2500))#1*1000

    start_iter = 0
    start_epoch= 0
    rate       = 0
    if initial_checkpoint is not None:
        initial_optimizer = initial_checkpoint.replace('_model.pth','_optimizer.pth')
        if os.path.exists(initial_optimizer):
            checkpoint  = torch.load(initial_optimizer)
            start_iter  = checkpoint['iter' ]
            start_epoch = checkpoint['epoch']

            optimizer.load_state_dict(checkpoint['optimizer'])
        pass



    log.write('optimizer\n  %s\n'%(optimizer))
    log.write('schduler\n  %s\n'%(schduler))
    log.write('\n')

    ## start training here! ##############################################

    log.write('** start training here! **\n')
    log.write('   batch_size =%d,  iter_accum=%d\n'%(batch_size,iter_accum))
    log.write('                      |--------------- VALID ----------------------------------------------------------------|-- TRAIN/BATCH ---------\n')
    log.write('                      |std %4.1f    %4.1f    %4.1f    %4.1f    %4.1f    %4.1f    %4.1f   %4.1f  |                    |        | \n'%tuple(COUPLING_TYPE_STD))
    log.write('rate     iter   epoch |    1JHC,   2JHC,   3JHC,   1JHN,   2JHN,   3JHN,   2JHH,   3JHH |  loss  mae log_mae | loss   | time          \n')
    log.write('--------------------------------------------------------------------------------------------------------------------------------------\n')
              #0.00100  111.0* 111.0 | 1.0 +1.2, 2.0 +1.2, 3.0 +1.2, 4.0 +1.2, 5.0 +1.2, 6.0 +1.2, 7.0 +1.2, 8.0 +1.2 | 8.01 +1.21  5.620 | 5.620 | 0 hr 04 min
               #    %5.2f     %5.2f     %5.2f     %5.2f     %5.2f     %5.2f     %5.2f     %5.2f

    train_loss   = np.zeros(20,np.float32)
    valid_loss   = np.zeros(20,np.float32)
    batch_loss   = np.zeros(20,np.float32)
    iter = 0
    i    = 0


    start = timer()
    while  iter<num_iters:
        sum_train_loss = np.zeros(20,np.float32)
        sum = 0

        optimizer.zero_grad()
        for node, edge, edge_index, node_index, coupling_value, coupling_index, infor in train_loader:

            #while 1:
                batch_size = len(infor)
                iter  = i + start_iter
                epoch = (iter-start_iter)*batch_size/len(train_dataset) + start_epoch


                # debug-----------------------------
                # if 0:
                #     pass

                #if 0:
                if (iter % iter_valid==0):
                    valid_loss = do_valid(net, valid_loader) #



                if (iter % iter_log==0):
                    print('\r',end='',flush=True)
                    asterisk = '*' if iter in iter_save else ' '
                    log.write('%0.5f  %5.1f%s %5.1f |  %+0.3f, %+0.3f, %+0.3f, %+0.3f, %+0.3f, %+0.3f, %+0.3f, %+0.3f | %+5.3f %5.2f %+0.2f | %+5.3f | %s' % (\
                             rate, iter/1000, asterisk, epoch,
                             *valid_loss[:11],
                             train_loss[0],
                             time_to_str((timer() - start),'min'))
                    )
                    log.write('\n')


                #if 0:
                if iter in iter_save:
                    torch.save(net.state_dict(),out_dir +'/checkpoint/%08d_model.pth'%(iter))
                    torch.save({
                        'optimizer': optimizer.state_dict(),
                        'iter'     : iter,
                        'epoch'    : epoch,
                    }, out_dir +'/checkpoint/%08d_optimizer.pth'%(iter))
                    pass




                # learning rate schduler -------------
                lr = schduler(iter)
                if lr<0 : break
                adjust_learning_rate(optimizer, lr)
                rate = get_learning_rate(optimizer)

                # one iteration update  -------------
                #net.set_mode('train',is_freeze_bn=True)

                net.train()
                node = node.cuda()
                edge = edge.cuda()
                edge_index = edge_index.cuda()
                node_index = node_index.cuda()
                coupling_value = coupling_value.cuda()
                coupling_index = coupling_index.cuda()


                predict = net(node, edge, edge_index, node_index, coupling_index)
                loss = criterion(predict, coupling_value)

                (loss/iter_accum).backward()
                if (iter % iter_accum)==0:
                    optimizer.step()
                    optimizer.zero_grad()

                # print statistics  ------------
                batch_loss[:1] = [loss.item()]
                sum_train_loss += batch_loss
                sum += 1
                if iter%iter_smooth == 0:
                    train_loss = sum_train_loss/sum
                    sum_train_loss = np.zeros(20,np.float32)
                    sum = 0


                print('\r',end='',flush=True)
                asterisk = ' '
                print('%0.5f  %5.1f%s %5.1f |  %+0.3f, %+0.3f, %+0.3f, %+0.3f, %+0.3f, %+0.3f, %+0.3f, %+0.3f | %+5.3f %5.2f %+0.2f | %+5.3f | %s' % (\
                             rate, iter/1000, asterisk, epoch,
                             *valid_loss[:11],
                             batch_loss[0],
                             time_to_str((timer() - start),'min'))
                , end='',flush=True)
                i=i+1


        pass  #-- end of one data loader --
    pass #-- end of all iterations --

    log.write('\n')





def run_submit(initial_checkpoint, out_dir=RESULT_DIR):

    csv_file = out_dir +'/submit-%s-larger.csv'%(initial_checkpoint.split('/')[-1][:-4])

    ## setup  -----------------------------------------------------------------------------
    os.makedirs(out_dir +'/checkpoint', exist_ok=True)
    os.makedirs(out_dir +'/submit', exist_ok=True)
    os.makedirs(out_dir +'/backup', exist_ok=True)
    # backup_project_as_zip(PROJECT_PATH, out_dir +'/backup/code.submit.%s.zip'%IDENTIFIER)

    log = Logger()
    log.open(out_dir+'/log.submit.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('\t%s\n' % COMMON_STRING)
    log.write('\n')


    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')
    batch_size = 20 #*2 #280*2 #256*4 #128 #256 #512  #16 #32

    if 0:## <debug>
        test_dataset = ChampsDataset(
                    mode ='train',
                    csv  ='train',
                    #split='debug_split_by_mol.1000.npy',
                    split='valid_split_by_mol.5000.npy',
                    augment=None,
        )

    #------------
    if 1:
        test_dataset = ChampsDataset(
                    mode ='test',
                    csv  ='test',
                    #split='debug_split_by_mol.1000.npy',
                    split=None,
                    augment=None,
        )
    test_loader  = DataLoader(
                test_dataset,
                sampler     = SequentialSampler(test_dataset),
                #sampler     = RandomSampler(train_dataset),
                batch_size  = batch_size,
                drop_last   = False,
                num_workers = 0,
                pin_memory  = True,
                collate_fn  = null_collate
    )

    log.write('batch_size = %d\n'%(batch_size))
    log.write('test_dataset : \n%s\n'%(test_dataset))
    log.write('\n')

    ## net ----------------------------------------
    log.write('** net setting **\n')
    net = Net(node_dim=NODE_DIM,edge_dim=EDGE_DIM, num_target=NUM_TARGET).cuda()

    log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
    net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))


    log.write('%s\n'%(type(net)))
    log.write('\n')


    ## start testing here! ##############################################
    test_num = 0
    test_predict = []
    test_coupling_type  = []
    test_coupling_value = []
    test_id = []

    test_loss = 0

    start = timer()
    for b, (node, edge, edge_index, node_index, coupling_value, coupling_index, infor) in enumerate(test_loader):

        net.eval()
        with torch.no_grad():
            node = node.cuda()
            edge = edge.cuda()
            edge_index = edge_index.cuda()
            node_index = node_index.cuda()
            coupling_index = coupling_index.cuda()
            coupling_value = coupling_value.cuda()

            predict = net(node, edge, edge_index, node_index, coupling_index)
            loss = criterion(predict, coupling_value)

        #---
        batch_size = len(infor)
        test_id.extend(list(np.concatenate([infor[b][2] for b in range(batch_size)])))

        test_predict.append(predict.data.cpu().numpy())
        test_coupling_type.append(coupling_index[:,2].data.cpu().numpy())
        test_coupling_value.append(coupling_value.data.cpu().numpy())

        test_loss += loss.item()*batch_size
        test_num += batch_size


        print('\r %8d/%8d     %0.2f  %s'%(
            test_num, len(test_dataset),test_num/len(test_dataset),
              time_to_str(timer()-start,'min')),end='',flush=True)


        pass  #-- end of one data loader --
    assert(test_num == len(test_dataset))
    print('\n')

    id  = test_id
    predict  = np.concatenate(test_predict)
    if test_dataset.mode == 'test':
        df = pd.DataFrame(list(zip(id, predict)), columns =['id', 'scalar_coupling_constant'])
        df.to_csv(csv_file,index=False)

        log.write('id        = %d\n'%len(id))
        log.write('predict   = %d\n'%len(predict))
        log.write('csv_file  = %s\n'%csv_file)

    #-------------------------------------------------------------
    # for debug
    if test_dataset.mode == 'train':
        test_loss = test_loss/test_num

        coupling_value = np.concatenate(test_coupling_value)
        coupling_type  = np.concatenate(test_coupling_type).astype(np.int32)

        mae, log_mae = compute_kaggle_metric( predict, coupling_value, coupling_type,)

        for t in range(NUM_COUPLING_TYPE):
            log.write('\tcoupling_type = %s\n'%COUPLING_TYPE[t])
            log.write('\tmae     =  %f\n'%mae[t])
            log.write('\tlog_mae = %+f\n'%log_mae[t])
            log.write('\n')
        log.write('\n')


        log.write('-- final -------------\n')
        log.write('\ttest_loss = %+f\n'%test_loss)
        log.write('\tmae       =  %f\n'%np.mean(mae))
        log.write('\tlog_mae   = %+f\n'%np.mean(log_mae))
        log.write('\n')


''' 
 
split = debug_split_by_mol.1000.npy
  
** start training here! **
   batch_size =20,  iter_accum=1
                      |--------------- VALID ----------------------------------------------------------------|-- TRAIN/BATCH --------
                      |std 18.3     4.5     3.1    10.9     3.7     1.3     4.0    3.7  |                    |        | 
rate     iter   epoch |    1JHC,   2JHC,   3JHC,   1JHN,   2JHN,   3JHN,   2JHH,   3JHH |  loss  mae log_mae | loss   | time         
-------------------------------------------------------------------------------------------------------------------------------------
0.00000    0.0*   0.0 |  +4.556, +1.231, +1.373, +3.907, +1.387, +0.172, +2.398, +1.643 | +3.042 21.71 +2.08 | +0.000 |  0 hr 00 min
0.00100    0.5   10.0 |  +1.022, -0.027, +0.320, +0.928, -0.048, -0.514, +0.016, +0.736 | +0.432  1.54 +0.30 | +0.663 |  0 hr 01 min
0.00100    1.0   20.0 |  +0.501, -0.397, +0.102, +0.306, -0.453, -0.669, -0.185, +0.518 | +0.097  1.06 -0.03 | +0.179 |  0 hr 02 min
0.00100    1.5   30.0 |  +0.429, -0.566, -0.052, +0.377, -0.680, -0.809, -0.577, +0.143 | -0.090  0.90 -0.22 | +0.156 |  0 hr 03 min
0.00100    2.0   40.0 |  +0.199, -0.554, -0.233, +0.106, -0.574, -0.907, -0.624, -0.146 | -0.258  0.76 -0.34 | -0.239 |  0 hr 04 min
0.00100    2.5*  50.0 |  +0.171, -0.786, -0.332, +0.212, -0.893, -0.975, -0.800, -0.112 | -0.344  0.72 -0.44 | -0.229 |  0 hr 05 min
0.00100    3.0   60.0 |  +0.151, -0.850, -0.465, +0.264, -0.936, -1.097, -0.968, -0.476 | -0.464  0.66 -0.55 | -0.427 |  0 hr 06 min
0.00100    3.5   70.0 |  -0.058, -0.920, -0.548, +0.168, -0.708, -1.374, -0.545, -0.501 | -0.524  0.63 -0.56 | -0.524 |  0 hr 07 min
0.00100    4.0   80.0 |  -0.047, -0.976, -0.537, -0.087, -0.992, -1.296, -1.079, -0.671 | -0.609  0.54 -0.71 | -0.621 |  0 hr 09 min
0.00100    4.5   90.0 |  -0.116, -0.864, -0.627, -0.249, -1.129, -1.514, -1.138, -0.624 | -0.645  0.50 -0.78 | -0.435 |  0 hr 10 min
0.00100    5.0* 100.0 |  -0.286, -0.972, -0.756, -0.240, -1.344, -1.705, -1.146, -0.787 | -0.777  0.45 -0.90 | -0.462 |  0 hr 11 min
0.00100    5.5  110.0 |  -0.388, -1.097, -0.657, -0.277, -1.151, -1.514, -1.234, -0.859 | -0.806  0.44 -0.90 | -0.693 |  0 hr 12 min
0.00100    6.0  120.0 |  -0.344, -0.996, -0.816, -0.367, -1.389, -1.614, -1.181, -0.880 | -0.833  0.42 -0.95 | -0.623 |  0 hr 13 min
0.00100    6.5  130.0 |  -0.360, -1.207, -0.739, -0.347, -1.249, -1.653, -1.201, -0.911 | -0.860  0.42 -0.96 | -0.656 |  0 hr 14 min
0.00100    7.0  140.0 |  -0.218, -1.106, -0.826, -0.611, -1.219, -1.652, -1.335, -1.042 | -0.849  0.40 -1.00 | -0.812 |  0 hr 15 min
0.00100    7.5* 150.0 |  -0.288, -1.312, -0.909, -0.597, -1.482, -1.751, -1.159, -1.121 | -0.940  0.38 -1.08 | -0.696 |  0 hr 16 min

'''
# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    # model = \
    #     '/home/alexpartisan/Work/Data/Competition/CHAMP/GNN/results/0703_2/checkpoint/00062500_model.pth'

    # model = \
    #     '/home/alexpartisan/Work/Data/Competition/CHAMP/GNN/results/0704/checkpoint/00075000_model.pth'

    # model = \
    #     '/home/alexpartisan/Work/Data/Competition/CHAMP/GNN/results/0708/checkpoint/00172500_model.pth'

    model = \
        '/home/alexpartisan/Work/Data/Competition/CHAMP/GNN/results/0709/checkpoint/00187500_model.pth'

    # run_train(out_dir=RES_DIR+'/0709', initial_checkpoint=model)

    run_submit(model, out_dir=RESULT_DIR)
