from lib.common import *
from data import *

from dscribe.descriptors import ACSF
from dscribe.core.system import System

ACSF_GENERATOR = ACSF(
    species = SYMBOL,
    rcut = 6.0,
    g2_params=[[1, 1], [1, 2], [1, 3]],
    g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]],
)




EDGE_DIM   =  6
NODE_DIM   = 93 ##  93  13
NUM_TARGET =  8

DATA_DIR = '/root/share/project/kaggle/2019/champs_scalar/data'

class ChampsDataset(Dataset):
    def __init__(self, split, csv, mode, augment=None):

        self.split   = split
        self.csv     = csv
        self.mode    = mode
        self.augment = augment

        self.df = pd.read_csv(DATA_DIR + '/csv/%s.csv'%csv)

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
        graph_file = DATA_DIR + '/structure/graph1/%s.pickle'%molecule_name
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



##############################################################

def run_check_train_dataset():

    dataset = ChampsDataset(
        mode ='train',
        csv  ='train',
        split='debug_split_by_mol.1000.npy',
        augment=None,
    )
    print(dataset)

    for n in range(0,len(dataset)):
        i = n
        #i = np.random.choice(len(dataset))

        graph = dataset[i]
        print(graph)
        print('graph.molecule_name:', graph.molecule_name)
        print('graph.smiles:', graph.smiles)
        print('-----')
        print('graph.node:', graph.node.shape)
        print('graph.edge:', graph.edge.shape)
        print('graph.edge_index:', graph.edge_index.shape)
        print('-----')
        print('graph.coupling.index:', graph.coupling.index.shape)
        print('graph.coupling.type:', graph.coupling.type.shape)
        print('graph.coupling.value:', graph.coupling.value.shape)
        print('graph.coupling.contribution:', graph.coupling.contribution.shape)
        print('graph.coupling.id:', graph.coupling.id)
        print('graph.coupling.type:', graph.coupling.type)
        print('')


        if 1:
            mol   = Chem.AddHs(Chem.MolFromSmiles(graph.smiles))
            image = np.array(Chem.Draw.MolToImage(mol,size=(128,128)))
            image_show('',image)
            cv2.waitKey(0)




def run_check_data_loader():


    dataset = ChampsDataset(
        mode ='train',
        csv  ='train',
        split='debug_split_by_mol.1000.npy',
        augment=None,
    )
    print(dataset)
    loader  = DataLoader(
            dataset,
            sampler     = SequentialSampler(dataset),
            #sampler     = RandomSampler(dataset),
            batch_size  = 32,
            drop_last   = False,
            num_workers = 4,
            pin_memory  = True,
            collate_fn  = null_collate)

    for b,(node, edge, edge_index, node_index, coupling_value, coupling_index, infor) in enumerate(loader):

        print('----b=%d---'%b)
        print('')
        print(infor)
        print(node.shape)
        print(edge.shape)
        print(edge_index.shape)
        print(node_index.shape)
        print('')
        print(coupling_value.shape)
        print(coupling_index.shape)

        print('')

# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    #run_check_train_dataset()
    run_check_data_loader()


    print('\nsucess!')