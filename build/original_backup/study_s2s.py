from common import *
import torch_geometric.nn as gnn
from torch_geometric.utils import scatter_
from torch_geometric.utils import add_self_loops, degree

from torch_geometric.nn import knn_graph
from torch_scatter import scatter_max, scatter_add
## https://github.com/JiaxuanYou/graph-pooling/blob/master/set2set.py


"""The global pooling operator based on iterative content-based attention
from the `"Order Matters: Sequence to sequence for sets"
<https://arxiv.org/abs/1511.06391>`_ paper

.. math::
    \mathbf{q}_t &= \mathrm{LSTM}(\mathbf{q}^{*}_{t-1})

    \alpha_{i,t} &= \mathrm{softmax}(\mathbf{x}_i \cdot \mathbf{q}_t)

    \mathbf{r}_t &= \sum_{i=1}^N \alpha_{i,t} \mathbf{x}_i

    \mathbf{q}^{*}_t &= \mathbf{q}_t \, \Vert \, \mathbf{r}_t,

where :math:`\mathbf{q}^{*}_T` defines the output of the layer with twice
the dimensionality as the input.

Args:
    in_channels (int): Size of each input sample.
    processing_steps (int): Number of iterations :math:`T`.
    num_layers (int, optional): Number of recurrent layers, *.e.g*, setting
        :obj:`num_layers=2` would mean stacking two LSTMs together to form
        a stacked LSTM, with the second LSTM taking in outputs of the first
        LSTM and computing the final results. (default: :obj:`1`)
"""

# def maybe_num_nodes(index, num_nodes=None):
#     return index.max().item() + 1 if num_nodes is None else num_nodes


def softmax(x, index, num=None):
    x = x -  scatter_max(x, index, dim=0, dim_size=num)[0][index]
    x = x.exp()
    x = x / (scatter_add(x, index, dim=0, dim_size=num)[index] + 1e-16)
    return x



class Set2Set(torch.nn.Module):


    def __init__(self, in_channel, processing_step=1):
        super(Set2Set, self).__init__()
        num_layer = 1
        out_channel = 2 * in_channel

        self.processing_step = processing_step
        self.in_channel  = in_channel
        self.out_channel = out_channel
        self.num_layer   = num_layer

        self.lstm = torch.nn.LSTM(out_channel, in_channel, num_layer)


    def forward(self, x, batch_index):
        batch_size = batch_index.max().item() + 1

        h = (x.new_zeros((self.num_layer, batch_size, self.in_channel)),
             x.new_zeros((self.num_layer, batch_size, self.in_channel)))

        q_star = x.new_zeros(batch_size, self.out_channel)

        for i in range(self.processing_step):
            q, h = self.lstm(q_star.unsqueeze(0), h)
            q = q.view(batch_size, -1)

            e = (x * q[batch_index]).sum(dim=-1, keepdim=True) #shape = num_node x 1
            a = softmax(e, batch_index, num=batch_size)        #shape = num_node x 1
            r = scatter_add(a * x, batch_index, dim=0, dim_size=batch_size) #apply attention #shape = batch_size x ...
            q_star = torch.cat([q, r], dim=-1)

        return q_star








##----
def run_check1():
    batch_size=10
    num_node=30
    node_dim=3
    edge_dim=2


    num_edge = 24
    edge_index = np.random.choice(num_node,(num_edge,2))
    node = np.random.uniform(-1,1,(num_node,node_dim))
    edge = np.random.uniform(-1,1,(num_edge,edge_dim))
    node_batch_index = np.random.choice(batch_size,num_node)
    node_batch_index = np.sort(node_batch_index)

    #---
    edge_index = torch.from_numpy(edge_index).long()
    node_batch_index = torch.from_numpy(node_batch_index).long()
    node = torch.from_numpy(node).float()
    edge = torch.from_numpy(edge).float()


    set2set_ref = gnn.Set2Set(node_dim, processing_steps=1)
    set2set = Set2Set(node_dim, processing_step=1)

    set2set.lstm.bias_ih_l0.data = set2set_ref.lstm.bias_ih_l0
    set2set.lstm.bias_hh_l0.data = set2set_ref.lstm.bias_hh_l0
    set2set.lstm.weight_ih_l0.data = set2set_ref.lstm.weight_ih_l0
    set2set.lstm.weight_hh_l0.data = set2set_ref.lstm.weight_hh_l0



    #---
    print('------------------------------')
    print('')
    print(set2set_ref)
    print('')

    print('node (x)')
    print(node.shape)
    print('')

    y  = set2set_ref(node, node_batch_index)
    y1 = set2set(node, node_batch_index)

    print('y')
    print(y.shape)
    print(y)
    print('')

    print('y1')
    print(y1.shape)
    print(y1)
    print('')




# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))
    run_check1()


    print('\nsucess!')
