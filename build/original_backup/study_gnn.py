from common import *
import torch_geometric.nn as gnn
from torch_geometric.utils import scatter_
from torch_geometric.utils import add_self_loops, degree

from torch_geometric.nn import knn_graph

## try to write our own graph layer
## https://rusty1s.github.io/pytorch_geometric/build/html/notes/create_gnn.html
##------------------------------
## https://rusty1s.github.io/pytorch_geometric/build/html/notes/create_gnn.html#implementing-the-edge-convolution


## https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html#module-torch_geometric.nn.conv.message_passing

class Identity(nn.Module):
    def __init__(self,):
        super(Identity, self).__init__()
    def forward(self, x):
        return x


class GConv(nn.Module):
    def __init__(self, node_dim, edge_dim, reduction, is_root, is_bias):
        super(GConv, self).__init__()
        self.reduction = reduction

        if is_bias:
            self.bias = nn.Parameter(torch.Tensor(node_dim))
        else:
            self.register_parameter('bias', None)

        if is_root:
            self.root = nn.Parameter(torch.Tensor(node_dim,node_dim))
        else:
            self.register_parameter('root', None)


    def forward(self, node, edge_index, edge):
        num_node, node_dim = node.shape
        num_edge, edge_dim = edge.shape

        #propagte:----

        #1. message :  m_j = SUM_i f(n_i, n_j, e_ij)  where i is neighbour(j)
        x_i     = torch.index_select(node, 0, edge_index[0])
        edge    = edge.view(-1,node_dim,node_dim)
        message = x_i.view(-1,node_dim,1)*edge
        message = message.sum(1)
        message = scatter_(self.reduction, message, edge_index[1], dim_size=num_node)

        if self.bias is not None:
            message += self.bias

        if self.root is not None:
            message = message + node@self.root

        #print('message')
        #print(message.shape)

        #2. update: n_j = f(n_j, m_j)
        update = message

        return update




def check_gconv(node, edge_index, edge, reduction, root_weight, bias):
    num_node, node_dim = node.shape
    num_edge = len(edge_index)


    if reduction=='mean' or reduction=='add':
        message = np.zeros((num_node,node_dim))
    if reduction=='max':
        message = np.ones((num_node,node_dim))*(-INF)

    count = np.zeros(num_node)
    for e,(i,j) in zip(edge, edge_index):

        x_i  = node[i].reshape((node_dim,1))
        e_ij = e.reshape(node_dim,node_dim)
        s = (x_i*e_ij).sum(0)

        if reduction=='mean' or reduction=='add':
            message[j] += s
            count[j] += 1
        if reduction=='max':
            message[j] = np.maximum(message[j],s)

    if reduction=='mean':
        message = message/(count.reshape(-1,1)+ 1e-8)

    if bias is not None:
        message += bias

    if root_weight is not None:
        message = message + node@root_weight

    update = message
    return update
'''
>>> s
array([[-0.08901411, -0.4506132 ,  0.14907332],
       [ 0.14621627, -0.20332962,  0.22908127],
       [-0.4580212 , -0.33847612, -0.5558714 ]], dtype=float32)
>>> s.sum(0)
array([-0.40081903, -0.99241894, -0.17771685], dtype=float32)
'''


def run_study_1():


    num_node=7
    node_dim=3
    edge_dim=node_dim*node_dim


    num_edge = 24
    edge_index = np.random.choice(num_node,(num_edge,2))
    node = np.random.uniform(-1,1,(num_node,node_dim))
    edge = np.random.uniform(-1,1,(num_edge,edge_dim))

    #---
    edge_index = torch.from_numpy(edge_index).long()
    node = torch.from_numpy(node).float()
    edge = torch.from_numpy(edge).float()

    aggr = 'mean'
    root_weight = True
    bias = True #False  True
    #gconv = gnn.NNConv(node_dim, node_dim, Identity(), aggr, root_weight, bias)

    gconv = GConv( node_dim, edge_dim, reduction=aggr, is_root=root_weight, is_bias=bias)


    #---
    print('------------------------------')
    print('')
    print(gconv)
    print('')

    print('node (x)')
    print(node.shape)
    print('')

    y = gconv(node, edge_index.t().contiguous(), edge)

    print('y')
    print(y.shape)
    print('')
    #exit(0)

    ##--------------------
    node       = node.data.cpu().numpy()
    edge_index = edge_index.data.cpu().numpy()
    edge       = edge.data.cpu().numpy()

    if bias == True:
        bias = gconv.bias.data.cpu().numpy()
    else:
        bias = None

    if root_weight == True:
        root_weight = gconv.root.data.cpu().numpy()
    else:
        root_weight = None

    my_y = check_gconv(node, edge_index, edge, reduction=aggr, root_weight=root_weight, bias=bias)


    ##---------------------------------
    print('my_y')
    print(my_y.shape)
    print(my_y)
    print('')
    print('---')

    print('y')
    print(y.shape)
    print(y)
    print('')
    print('---')


# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))


    run_study_1()

    print('\nsucess!')
