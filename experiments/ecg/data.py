import matplotlib.pyplot as plt
#import librosa, librosa.display
import os
import numpy as np
import torch
from  torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split
#import librosa
import random
from matplotlib import pyplot as plt
import copy

np.random.seed(42)

def normalize(seq):
    '''
    normalize to [-1,1]
    :param seq:
    :return:
    '''
    return 2*(seq-np.min(seq))/(np.max(seq)-np.min(seq))-1
    #return (seq-np.min(seq))/(np.max(seq)-np.min(seq))    
    #return librosa.util.normalize(seq,axis=1)


class MultimodalDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


def load_data(opt):
    train_dataset=None
    test_dataset=None
    val_dataset=None
    test_N_dataset=None
    test_S_dataset = None
    test_V_dataset = None
    test_F_dataset = None
    test_Q_dataset = None

    if opt.dataset=="ecg":

        #- signal
        N_samples_s = np.load(os.path.join(opt.dataroot, "N_samples.npy")) #NxCxL
        S_samples_s = np.load(os.path.join(opt.dataroot, "S_samples.npy"))
        V_samples_s = np.load(os.path.join(opt.dataroot, "V_samples.npy"))
        F_samples_s = np.load(os.path.join(opt.dataroot, "F_samples.npy"))
        Q_samples_s = np.load(os.path.join(opt.dataroot, "Q_samples.npy"))
        
        #- freq
        N_samples_f = np.load(os.path.join(opt.dataroot,'n_spectrogram.npy'))
        S_samples_f = np.load(os.path.join(opt.dataroot,'s_spectrogram.npy'))
        V_samples_f = np.load(os.path.join(opt.dataroot,'v_spectrogram.npy'))
        F_samples_f = np.load(os.path.join(opt.dataroot,'f_spectrogram.npy'))
        Q_samples_f = np.load(os.path.join(opt.dataroot,'q_spectrogram.npy'))
        
        #N_samples_s = N_samples_s[0:100]
        #N_samples_f = N_samples_f[0:100]


        # normalize signal
        for i in range(N_samples_s.shape[0]):
            for j in range(opt.nc):
                N_samples_s[i][j]=normalize(N_samples_s[i][j][:])
        N_samples_s=N_samples_s[:,:opt.nc,:]

        for i in range(S_samples_s.shape[0]):
            for j in range(opt.nc):
                S_samples_s[i][j] = normalize(S_samples_s[i][j][:])
        S_samples_s = S_samples_s[:, :opt.nc, :]

        for i in range(V_samples_s.shape[0]):
            for j in range(opt.nc):
                V_samples_s[i][j] = normalize(V_samples_s[i][j][:])
        V_samples_s = V_samples_s[:, :opt.nc, :]

        for i in range(F_samples_s.shape[0]):
            for j in range(opt.nc):
                F_samples_s[i][j] = normalize(F_samples_s[i][j][:])
        F_samples_s = F_samples_s[:, :opt.nc, :]

        for i in range(Q_samples_s.shape[0]):
            for j in range(opt.nc):
                Q_samples_s[i][j] = normalize(Q_samples_s[i][j][:])
        Q_samples_s = Q_samples_s[:, :opt.nc, :]
        
        # normalize freq
        for i in range(N_samples_f.shape[0]):
            N_samples_f[i] = normalize(N_samples_f[i])
                    
        for i in range(S_samples_f.shape[0]):
            S_samples_f[i] = normalize(S_samples_f[i])

        for i in range(V_samples_f.shape[0]):
            V_samples_f[i] = normalize(V_samples_f[i])
            
        for i in range(F_samples_f.shape[0]):
            F_samples_f[i] = normalize(F_samples_f[i])

        for i in range(Q_samples_f.shape[0]):
            Q_samples_f[i] = normalize(Q_samples_f[i])    
    


        # train / test
        #- signal
        test_N_s, test_N_y_s, train_N_s, train_N_y_s = getFloderK(N_samples_s,opt.folder,0)
        test_S_s, test_S_y_s = S_samples_s, np.ones((S_samples_s.shape[0], 1))
        test_V_s, test_V_y_s = V_samples_s, np.ones((V_samples_s.shape[0], 1))
        test_F_s, test_F_y_s = F_samples_s, np.ones((F_samples_s.shape[0], 1))
        test_Q_s, test_Q_y_s = Q_samples_s, np.ones((Q_samples_s.shape[0], 1))
        
        #- freq
        test_N_f, test_N_y_f, train_N_f, train_N_y_f = getFloderK(N_samples_f,opt.folder,0)
        test_S_f, test_S_y_f = S_samples_f, np.ones((S_samples_f.shape[0], 1))
        test_V_f, test_V_y_f = V_samples_f, np.ones((V_samples_f.shape[0], 1))
        test_F_f, test_F_y_f = F_samples_f, np.ones((F_samples_f.shape[0], 1))
        test_Q_f, test_Q_y_f = Q_samples_f, np.ones((Q_samples_f.shape[0], 1))        


        # train / val
        #- signal
        train_N_s, val_N_s, train_N_y_s, val_N_y_s = getPercent(train_N_s, train_N_y_s, 0.1, 0)
        test_S_s, val_S_s, test_S_y_s, val_S_y_s = getPercent(test_S_s, test_S_y_s, 0.1, 0)
        test_V_s, val_V_s, test_V_y_s, val_V_y_s = getPercent(test_V_s, test_V_y_s, 0.1, 0)
        test_F_s, val_F_s, test_F_y_s, val_F_y_s = getPercent(test_F_s, test_F_y_s, 0.1, 0)
        test_Q_s, val_Q_s, test_Q_y_s, val_Q_y_s = getPercent(test_Q_s, test_Q_y_s, 0.1, 0)

        val_data_s=np.concatenate([val_N_s,val_S_s,val_V_s,val_F_s,val_Q_s])
        val_y_s=np.concatenate([val_N_y_s,val_S_y_s,val_V_y_s,val_F_y_s,val_Q_y_s])
        
        #- freq
        train_N_f, val_N_f, train_N_y_f, val_N_y_f = getPercent(train_N_f, train_N_y_f, 0.1, 0)
        test_S_f, val_S_f, test_S_y_f, val_S_y_f = getPercent(test_S_f, test_S_y_f, 0.1, 0)
        test_V_f, val_V_f, test_V_y_f, val_V_y_f = getPercent(test_V_f, test_V_y_f, 0.1, 0)
        test_F_f, val_F_f, test_F_y_f, val_F_y_f = getPercent(test_F_f, test_F_y_f, 0.1, 0)
        test_Q_f, val_Q_f, test_Q_y_f, val_Q_y_f = getPercent(test_Q_f, test_Q_y_f, 0.1, 0)

        val_data_f=np.concatenate([val_N_f,val_S_f,val_V_f,val_F_f,val_Q_f])
        val_y_f=np.concatenate([val_N_y_f,val_S_y_f,val_V_y_f,val_F_y_f,val_Q_y_f])


        # train_N = np.stack((train_N_s, train_N_f), axis=0)
        # train_N_y = np.stack((train_N_y_s, train_N_y_f), axis=0)
        # test_N = np.stack((test_N_s, test_N_f), axis=0)

        print("\n############ signal dataset ############")
        print("train_s data size:{}".format(train_N_s.shape))
        print("val_s data size:{}".format(val_data_s.shape))
        print("test_s N data size:{}".format(test_N_s.shape))
        print("test_s S data size:{}".format(test_S_s.shape))
        print("test_s V data size:{}".format(test_V_s.shape))
        print("test_s F data size:{}".format(test_F_s.shape))
        print("test_s Q data size:{}".format(test_Q_s.shape))

        print("\n############ frequency dataset ############")
        print("train_f data size:{}".format(train_N_f.shape))
        print("val_f data size:{}".format(val_data_f.shape))
        print("test_f N data size:{}".format(test_N_f.shape))
        print("test_f S data size:{}".format(test_S_f.shape))
        print("test_f V data size:{}".format(test_V_f.shape))
        print("test_f F data size:{}".format(test_F_f.shape))
        print("test_f Q data size:{}".format(test_Q_f.shape))


        # if not opt.istest and opt.n_aug>0:
        #     train_N,train_N_y=data_aug(train_N,train_N_y,times=opt.n_aug)
        #     print("after aug, train data size:{}".format(train_N.shape))


        train_dataset_s  = TensorDataset(torch.Tensor(train_N_s),torch.Tensor(train_N_y_s))
        val_dataset_s    = TensorDataset(torch.Tensor(val_data_s), torch.Tensor(val_y_s))
        test_N_dataset_s = TensorDataset(torch.Tensor(test_N_s), torch.Tensor(test_N_y_s))
        test_S_dataset_s = TensorDataset(torch.Tensor(test_S_s), torch.Tensor(test_S_y_s))
        test_V_dataset_s = TensorDataset(torch.Tensor(test_V_s), torch.Tensor(test_V_y_s))
        test_F_dataset_s = TensorDataset(torch.Tensor(test_F_s), torch.Tensor(test_F_y_s))
        test_Q_dataset_s = TensorDataset(torch.Tensor(test_Q_s), torch.Tensor(test_Q_y_s))

        train_dataset_f  = TensorDataset(torch.Tensor(train_N_f),torch.Tensor(train_N_y_f))
        val_dataset_f    = TensorDataset(torch.Tensor(val_data_f), torch.Tensor(val_y_f))
        test_N_dataset_f = TensorDataset(torch.Tensor(test_N_f), torch.Tensor(test_N_y_f))
        test_S_dataset_f = TensorDataset(torch.Tensor(test_S_f), torch.Tensor(test_S_y_f))
        test_V_dataset_f = TensorDataset(torch.Tensor(test_V_f), torch.Tensor(test_V_y_f))
        test_F_dataset_f = TensorDataset(torch.Tensor(test_F_f), torch.Tensor(test_F_y_f))
        test_Q_dataset_f = TensorDataset(torch.Tensor(test_Q_f), torch.Tensor(test_Q_y_f))


    # assert (train_dataset is not None  and test_dataset is not None and val_dataset is not None)

    dataloader = {"train": DataLoader(#------------------------------------------------signal
                        dataset=MultimodalDataset(train_dataset_s, train_dataset_f),  # torch TensorDataset format
                        batch_size=opt.batchsize,  # mini batch size
                        shuffle=True,
                        num_workers=int(opt.workers),
                        drop_last=True),
                    "val": DataLoader(
                        dataset=MultimodalDataset(val_dataset_s, val_dataset_f),  # torch TensorDataset format
                        batch_size=opt.batchsize,  # mini batch size
                        shuffle=True,
                        num_workers=int(opt.workers),
                        drop_last=False),
                    "test_N":DataLoader(
                            dataset=MultimodalDataset(test_N_dataset_s, test_N_dataset_f),  # torch TensorDataset format
                            batch_size=opt.batchsize,  # mini batch size
                            shuffle=True,
                            num_workers=int(opt.workers),
                            drop_last=False),
                    "test_S": DataLoader(
                        dataset=MultimodalDataset(test_S_dataset_s, test_S_dataset_f),  # torch TensorDataset format
                        batch_size=opt.batchsize,  # mini batch size
                        shuffle=True,
                        num_workers=int(opt.workers),
                        drop_last=False),
                    "test_V": DataLoader(
                        dataset=MultimodalDataset(test_V_dataset_s, test_V_dataset_f),  # torch TensorDataset format
                        batch_size=opt.batchsize,  # mini batch size
                        shuffle=True,
                        num_workers=int(opt.workers),
                        drop_last=False),
                    "test_F": DataLoader(
                        dataset=MultimodalDataset(test_F_dataset_s, test_F_dataset_f),  # torch TensorDataset format
                        batch_size=opt.batchsize,  # mini batch size
                        shuffle=True,
                        num_workers=int(opt.workers),
                        drop_last=False),
                    "test_Q": DataLoader(
                        dataset=MultimodalDataset(test_Q_dataset_s, test_Q_dataset_f),  # torch TensorDataset format
                        batch_size=opt.batchsize,  # mini batch size
                        shuffle=True,
                        num_workers=int(opt.workers),
                        drop_last=False),
                    }
    return dataloader


def getFloderK(data,folder,label):
    normal_cnt = data.shape[0]
    folder_num = int(normal_cnt / 5)
    folder_idx = folder * folder_num

    folder_data = data[folder_idx:folder_idx + folder_num]

    remain_data = np.concatenate([data[:folder_idx], data[folder_idx + folder_num:]])
    if label==0:
        folder_data_y = np.zeros((folder_data.shape[0], 1))
        remain_data_y=np.zeros((remain_data.shape[0], 1))
    elif label==1:
        folder_data_y = np.ones((folder_data.shape[0], 1))
        remain_data_y = np.ones((remain_data.shape[0], 1))
    else:
        raise Exception("label should be 0 or 1, get:{}".format(label))
    return folder_data,folder_data_y,remain_data,remain_data_y

def getPercent(data_x,data_y,percent,seed):
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y,test_size=percent,random_state=seed)
    return train_x, test_x, train_y, test_y

def get_full_data(dataloader):

    full_data_x=[]
    full_data_y=[]
    for batch_data in dataloader:
        batch_x,batch_y=batch_data[0],batch_data[1]
        batch_x=batch_x.numpy()
        batch_y=batch_y.numpy()

        # print(batch_x.shape)
        # assert False
        for i in range(batch_x.shape[0]):
            full_data_x.append(batch_x[i,0,:])
            full_data_y.append(batch_y[i])

    full_data_x=np.array(full_data_x)
    full_data_y=np.array(full_data_y)
    assert full_data_x.shape[0]==full_data_y.shape[0]
    print("full data size:{}".format(full_data_x.shape))
    return full_data_x,full_data_y


def data_aug(train_x,train_y,times=2):
    #fig = plt.figure(figsize=(5,5))
    res_train_x=[]
    res_train_y=[]
    for idx in range(train_x.shape[0]):
        x=train_x[idx]
        y=train_y[idx]
        res_train_x.append(x)
        res_train_y.append(y)

        for i in range(times):
            x_aug=aug_ts(x)
            res_train_x.append(x_aug)
            res_train_y.append(y)
            '''
            img = librosa.display.specshow(x_aug[0], sr=360, hop_length = 2, y_axis="linear", x_axis="time")
            fig.savefig("aug/aug{0}.png".format(idx))
            img = librosa.display.specshow(x[0], sr=360, hop_length = 2, y_axis="linear", x_axis="time")
            fig.savefig("aug/real{0}.png".format(idx))
            '''
    res_train_x=np.array(res_train_x)
    res_train_y=np.array(res_train_y)

    return res_train_x,res_train_y




def aug_ts(feat , T = 5, F = 5, time_mask_num = 1, freq_mask_num = 1):
    feat1 = copy.deepcopy(feat)
    feat1_size = 128
    seq_len = 128

    # time mask
    for _ in range(time_mask_num):
        t = np.random.uniform(low=0.0, high=T)
        t = int(t)
        t0 = random.randint(1, seq_len - t) 
        feat1[:, :,t0 : t0 + t] = 0
        

    # freq mask
    for _ in range(freq_mask_num):
        f = np.random.uniform(low=0.0, high=F)
        f = int(f)
        f0 = random.randint(1, feat1_size - f - 100)
        feat1[:, f0 : f0 + f] = 0
        print(feat1.shape)

    return feat1
   
'''
def aug_ts(x):
    #x[0] = time_warp(x[0])
    print(time_warp(x).shape)
    return x
    
import torch
'''
#Export
def sparse_image_warp(img_tensor,
                      source_control_point_locations,
                      dest_control_point_locations,
                      interpolation_order=2,
                      regularization_weight=0.0,
                      num_boundaries_points=0):
    device = img_tensor.device
    control_point_flows = (dest_control_point_locations - source_control_point_locations)   
    
#     clamp_boundaries = num_boundary_points > 0
#     boundary_points_per_edge = num_boundary_points - 1
    batch_size, image_height, image_width = img_tensor.shape
    flattened_grid_locations = get_flat_grid_locations(image_height, image_width, device)

    # IGNORED FOR OUR BASIC VERSION...
#     flattened_grid_locations = constant_op.constant(
#         _expand_to_minibatch(flattened_grid_locations, batch_size), image.dtype)

#     if clamp_boundaries:
#       (dest_control_point_locations,
#        control_point_flows) = _add_zero_flow_controls_at_boundary(
#            dest_control_point_locations, control_point_flows, image_height,
#            image_width, boundary_points_per_edge)

    flattened_flows = interpolate_spline(
        dest_control_point_locations,
        control_point_flows,
        flattened_grid_locations,
        interpolation_order,
        regularization_weight)

    dense_flows = create_dense_flows(flattened_flows, batch_size, image_height, image_width)

    warped_image = dense_image_warp(img_tensor, dense_flows)

    return warped_image, dense_flows
    
#Export
def get_grid_locations(image_height, image_width, device):
    y_range = torch.linspace(0, image_height - 1, image_height, device=device)
    x_range = torch.linspace(0, image_width - 1, image_width, device=device)
    y_grid, x_grid = torch.meshgrid(y_range, x_range)
    return torch.stack((y_grid, x_grid), -1)
    
#Export
def flatten_grid_locations(grid_locations, image_height, image_width):
    return torch.reshape(grid_locations, [image_height * image_width, 2])
    
    
#Export
def get_flat_grid_locations(image_height, image_width, device):
    y_range = torch.linspace(0, image_height - 1, image_height, device=device)
    x_range = torch.linspace(0, image_width - 1, image_width, device=device)
    y_grid, x_grid = torch.meshgrid(y_range, x_range)
    return torch.stack((y_grid, x_grid), -1).reshape([image_height * image_width, 2])
    
#Export
def create_dense_flows(flattened_flows, batch_size, image_height, image_width):
    # possibly .view
    return torch.reshape(flattened_flows, [batch_size, image_height, image_width, 2])
    
#Export
def interpolate_spline(train_points, train_values, query_points, order, regularization_weight=0.0,):
    # First, fit the spline to the observed data.
    w, v = solve_interpolation(train_points, train_values, order, regularization_weight)
    # Then, evaluate the spline at the query locations.
    query_values = apply_interpolation(query_points, train_points, w, v, order)

    return query_values
    
#Export
def solve_interpolation(train_points, train_values, order, regularization_weight, eps=1e-7):
    device = train_points.device
    b, n, d = train_points.shape
    k = train_values.shape[-1]

    # First, rename variables so that the notation (c, f, w, v, A, B, etc.)
    # follows https://en.wikipedia.org/wiki/Polyharmonic_spline.
    # To account for python style guidelines we use
    # matrix_a for A and matrix_b for B.
    
    c = train_points
    f = train_values.float()
    
    matrix_a = phi(cross_squared_distance_matrix(c,c), order).unsqueeze(0)  # [b, n, n]
#     if regularization_weight > 0:
#         batch_identity_matrix = array_ops.expand_dims(
#           linalg_ops.eye(n, dtype=c.dtype), 0)
#         matrix_a += regularization_weight * batch_identity_matrix

    # Append ones to the feature values for the bias term in the linear model.
    ones = torch.ones(n, dtype=train_points.dtype, device=device).view([-1, n, 1])
    matrix_b = torch.cat((c, ones), 2).float()  # [b, n, d + 1]

    # [b, n + d + 1, n]
    left_block = torch.cat((matrix_a, torch.transpose(matrix_b, 2, 1)), 1)

    num_b_cols = matrix_b.shape[2]  # d + 1

    # In Tensorflow, zeros are used here. Pytorch solve fails with zeros for some reason we don't understand.
    # So instead we use very tiny randn values (variance of one, zero mean) on one side of our multiplication.
    lhs_zeros = torch.randn((b, num_b_cols, num_b_cols), device=device) *eps
    right_block = torch.cat((matrix_b, lhs_zeros),
                                   1)  # [b, n + d + 1, d + 1]
    lhs = torch.cat((left_block, right_block),
                           2)  # [b, n + d + 1, n + d + 1]

    rhs_zeros = torch.zeros((b, d + 1, k), dtype=train_points.dtype, device=device).float()
    rhs = torch.cat((f, rhs_zeros), 1)  # [b, n + d + 1, k]

    # Then, solve the linear system and unpack the results.
    X, LU = torch.solve(rhs, lhs)
    w = X[:, :n, :]
    v = X[:, n:, :]
    return w, v
#Export
def cross_squared_distance_matrix(x, y):
    """Pairwise squared distance between two (batch) matrices' rows (2nd dim).
        Computes the pairwise distances between rows of x and rows of y
        Args:
        x: [batch_size, n, d] float `Tensor`
        y: [batch_size, m, d] float `Tensor`
        Returns:
        squared_dists: [batch_size, n, m] float `Tensor`, where
        squared_dists[b,i,j] = ||x[b,i,:] - y[b,j,:]||^2
    """
    x_norm_squared = torch.sum(torch.mul(x, x))
    y_norm_squared = torch.sum(torch.mul(y, y))

    x_y_transpose = torch.matmul(x.squeeze(0), y.squeeze(0).transpose(0,1))
    
    # squared_dists[b,i,j] = ||x_bi - y_bj||^2 = x_bi'x_bi- 2x_bi'x_bj + x_bj'x_bj
    squared_dists = x_norm_squared - 2 * x_y_transpose + y_norm_squared

    return squared_dists.float()

#Export
def phi(r, order):
    """Coordinate-wise nonlinearity used to define the order of the interpolation.
    See https://en.wikipedia.org/wiki/Polyharmonic_spline for the definition.
    Args:
    r: input op
    order: interpolation order
    Returns:
    phi_k evaluated coordinate-wise on r, for k = r
    """
    EPSILON=torch.tensor(1e-10, device=r.device)
    # using EPSILON prevents log(0), sqrt0), etc.
    # sqrt(0) is well-defined, but its gradient is not
    if order == 1:
        r = torch.max(r, EPSILON)
        r = torch.sqrt(r)
        return r
    elif order == 2:
        return 0.5 * r * torch.log(torch.max(r, EPSILON))
    elif order == 4:
        return 0.5 * torch.square(r) * torch.log(torch.max(r, EPSILON))
    elif order % 2 == 0:
        r = torch.max(r, EPSILON)
        return 0.5 * torch.pow(r, 0.5 * order) * torch.log(r)
    else:
        r = torch.max(r, EPSILON)
        return torch.pow(r, 0.5 * order)

#Export
def apply_interpolation(query_points, train_points, w, v, order):
    """Apply polyharmonic interpolation model to data.
    Given coefficients w and v for the interpolation model, we evaluate
    interpolated function values at query_points.
    Args:
    query_points: `[b, m, d]` x values to evaluate the interpolation at
    train_points: `[b, n, d]` x values that act as the interpolation centers
                    ( the c variables in the wikipedia article)
    w: `[b, n, k]` weights on each interpolation center
    v: `[b, d, k]` weights on each input dimension
    order: order of the interpolation
    Returns:
    Polyharmonic interpolation evaluated at points defined in query_points.
    """
    query_points = query_points.unsqueeze(0)
    # First, compute the contribution from the rbf term.
    pairwise_dists = cross_squared_distance_matrix(query_points.float(), train_points.float())
    phi_pairwise_dists = phi(pairwise_dists, order)

    rbf_term = torch.matmul(phi_pairwise_dists, w)

    # Then, compute the contribution from the linear term.
    # Pad query_points with ones, for the bias term in the linear model.
    ones = torch.ones_like(query_points[..., :1])
    query_points_pad = torch.cat((
      query_points,
      ones
    ), 2).float()
    linear_term = torch.matmul(query_points_pad, v)

    return rbf_term + linear_term


#Export
def dense_image_warp(image, flow):
    """Image warping using per-pixel flow vectors.
    Apply a non-linear warp to the image, where the warp is specified by a dense
    flow field of offset vectors that define the correspondences of pixel values
    in the output image back to locations in the  source image. Specifically, the
    pixel value at output[b, j, i, c] is
    images[b, j - flow[b, j, i, 0], i - flow[b, j, i, 1], c].
    The locations specified by this formula do not necessarily map to an int
    index. Therefore, the pixel value is obtained by bilinear
    interpolation of the 4 nearest pixels around
    (b, j - flow[b, j, i, 0], i - flow[b, j, i, 1]). For locations outside
    of the image, we use the nearest pixel values at the image boundary.
    Args:
    image: 4-D float `Tensor` with shape `[batch, height, width, channels]`.
    flow: A 4-D float `Tensor` with shape `[batch, height, width, 2]`.
    name: A name for the operation (optional).
    Note that image and flow can be of type tf.half, tf.float32, or tf.float64,
    and do not necessarily have to be the same type.
    Returns:
    A 4-D float `Tensor` with shape`[batch, height, width, channels]`
    and same type as input image.
    Raises:
    ValueError: if height < 2 or width < 2 or the inputs have the wrong number
    of dimensions.
    """
    image = image.unsqueeze(3) # add a single channel dimension to image tensor
    batch_size, height, width, channels = image.shape
    device = image.device

    # The flow is defined on the image grid. Turn the flow into a list of query
    # points in the grid space.
    grid_x, grid_y = torch.meshgrid(
        torch.arange(width, device=device), torch.arange(height, device=device))
    
    stacked_grid = torch.stack((grid_y, grid_x), dim=2).float()
    
    batched_grid = stacked_grid.unsqueeze(-1).permute(3, 1, 0, 2)
    
    query_points_on_grid = batched_grid - flow
    query_points_flattened = torch.reshape(query_points_on_grid,
                                               [batch_size, height * width, 2])
    # Compute values at the query points, then reshape the result back to the
    # image grid.
    interpolated = interpolate_bilinear(image, query_points_flattened)
    interpolated = torch.reshape(interpolated,
                                     [batch_size, height, width, channels])
    return interpolated


#Export
def interpolate_bilinear(grid,
                         query_points,
                         name='interpolate_bilinear',
                         indexing='ij'):
    """Similar to Matlab's interp2 function.
    Finds values for query points on a grid using bilinear interpolation.
    Args:
    grid: a 4-D float `Tensor` of shape `[batch, height, width, channels]`.
    query_points: a 3-D float `Tensor` of N points with shape `[batch, N, 2]`.
    name: a name for the operation (optional).
    indexing: whether the query points are specified as row and column (ij),
      or Cartesian coordinates (xy).
    Returns:
    values: a 3-D `Tensor` with shape `[batch, N, channels]`
    Raises:
    ValueError: if the indexing mode is invalid, or if the shape of the inputs
      invalid.
    """
    if indexing != 'ij' and indexing != 'xy':
        raise ValueError('Indexing mode must be \'ij\' or \'xy\'')


    shape = grid.shape
    if len(shape) != 4:
      msg = 'Grid must be 4 dimensional. Received size: '
      raise ValueError(msg + str(grid.shape))

    batch_size, height, width, channels = grid.shape

    shape = [batch_size, height, width, channels]
    query_type = query_points.dtype
    grid_type = grid.dtype
    grid_device = grid.device

    num_queries = query_points.shape[1]

    alphas = []
    floors = []
    ceils = []
    index_order = [0, 1] if indexing == 'ij' else [1, 0]
    unstacked_query_points = query_points.unbind(2)
    
    for dim in index_order:
        queries = unstacked_query_points[dim]

        size_in_indexing_dimension = shape[dim + 1]

        # max_floor is size_in_indexing_dimension - 2 so that max_floor + 1
        # is still a valid index into the grid.
        max_floor = torch.tensor(size_in_indexing_dimension - 2, dtype=query_type, device=grid_device)
        min_floor = torch.tensor(0.0, dtype=query_type, device=grid_device)
        maxx = torch.max(min_floor, torch.floor(queries))
        floor = torch.min(maxx, max_floor)
        int_floor = floor.long()
        floors.append(int_floor)
        ceil = int_floor + 1
        ceils.append(ceil)

        # alpha has the same type as the grid, as we will directly use alpha
        # when taking linear combinations of pixel values from the image.
        
        
        alpha = (queries - floor).clone().detach().type(grid_type)
        min_alpha = torch.tensor(0.0, dtype=grid_type, device=grid_device)
        max_alpha = torch.tensor(1.0, dtype=grid_type, device=grid_device)
        alpha = torch.min(torch.max(min_alpha, alpha), max_alpha)

        # Expand alpha to [b, n, 1] so we can use broadcasting
        # (since the alpha values don't depend on the channel).
        alpha = torch.unsqueeze(alpha, 2)
        alphas.append(alpha)

    flattened_grid = torch.reshape(
      grid, [batch_size * height * width, channels])
    batch_offsets = torch.reshape(
      torch.arange(batch_size, device=grid_device) * height * width, [batch_size, 1])

    # This wraps array_ops.gather. We reshape the image data such that the
    # batch, y, and x coordinates are pulled into the first dimension.
    # Then we gather. Finally, we reshape the output back. It's possible this
    # code would be made simpler by using array_ops.gather_nd.
    def gather(y_coords, x_coords, name):
        linear_coordinates = batch_offsets + y_coords * width + x_coords
        gathered_values = torch.gather(flattened_grid.t(), 1, linear_coordinates)
        return torch.reshape(gathered_values,
                                 [batch_size, num_queries, channels])

    # grab the pixel values in the 4 corners around each query point
    top_left = gather(floors[0], floors[1], 'top_left')
    top_right = gather(floors[0], ceils[1], 'top_right')
    bottom_left = gather(ceils[0], floors[1], 'bottom_left')
    bottom_right = gather(ceils[0], ceils[1], 'bottom_right')
    
    interp_top = alphas[1] * (top_right - top_left) + top_left
    interp_bottom = alphas[1] * (bottom_right - bottom_left) + bottom_left
    interp = alphas[0] * (interp_bottom - interp_top) + interp_top

    return interp
    
#Export
def time_warp(spec, W=50):
    num_rows = spec.shape[2]
    spec_len = spec.shape[1]
    device = spec.device

    # adapted from https://github.com/DemisEom/SpecAugment/
    pt = (num_rows - 2* W) * torch.rand([1], dtype=torch.float) + W # random point along the time axis
    src_ctr_pt_freq = torch.arange(0, spec_len // 2)  # control points on freq-axis
    src_ctr_pt_time = torch.ones_like(src_ctr_pt_freq) * pt  # control points on time-axis
    src_ctr_pts = torch.stack((src_ctr_pt_freq, src_ctr_pt_time), dim=-1)
    src_ctr_pts = src_ctr_pts.float().to(device)

    # Destination
    w = 2 * W * torch.rand([1], dtype=torch.float) - W# distance
    dest_ctr_pt_freq = src_ctr_pt_freq
    dest_ctr_pt_time = src_ctr_pt_time + w
    dest_ctr_pts = torch.stack((dest_ctr_pt_freq, dest_ctr_pt_time), dim=-1)
    dest_ctr_pts = dest_ctr_pts.float().to(device)

    # warp
    source_control_point_locations = torch.unsqueeze(src_ctr_pts, 0)  # (1, v//2, 2)
    dest_control_point_locations = torch.unsqueeze(dest_ctr_pts, 0)  # (1, v//2, 2)
    warped_spectro, dense_flows = sparse_image_warp(spec, source_control_point_locations, dest_control_point_locations)
    return warped_spectro.squeeze(3)
    

'''
def aug_ts(x):
    left_ticks_index = np.arange(0, 140)
    right_ticks_index = np.arange(140, 319)
    np.random.shuffle(left_ticks_index)
    np.random.shuffle(right_ticks_index)
    left_up_ticks = left_ticks_index[:7]
    right_up_ticks = right_ticks_index[:7]
    left_down_ticks = left_ticks_index[7:14]
    right_down_ticks = right_ticks_index[7:14]

    x_1 = np.zeros_like(x)
    j = 0
    for i in range(x.shape[1]):
        if i in left_down_ticks or i in right_down_ticks:
            continue
        elif i in left_up_ticks or i in right_up_ticks:
            x_1[:, j] =x[:,i]
            j += 1
            x_1[:, j] = (x[:, i] + x[:, i + 1]) / 2
            j += 1
        else:
            x_1[:, j] = x[:, i]
            j += 1
    return x_1
'''


