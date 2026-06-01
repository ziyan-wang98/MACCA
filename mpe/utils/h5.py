
# --------------------------- hdf5 -------------------------------
import h5py
import torch as th
import numpy as np

def read_h5(filename):
    
    hdFile_r = h5py.File(filename, 'r')
    actions_h = th.tensor(np.array(hdFile_r.get('actions')))
    print("Finished loading actions, the shape is" + str(actions_h.shape) + ".")
    actions_onehot_h = th.tensor(np.array(hdFile_r.get('actions_onehot')))
    print("Finished loading actions one_hot, the shape is" + str(actions_onehot_h.shape) + ".")
    avail_actions_h = th.tensor(np.array(hdFile_r.get('avail_actions')))
    print("Finished loading avail actions , the shape is" + str(avail_actions_h.shape) + ".")
    filled_h = th.tensor(np.array(hdFile_r.get('filled')))
    print("Finished loading filled , the shape is" + str(filled_h.shape) + ".")
    obs_h = th.tensor(np.array(hdFile_r.get('obs')))
    print("Finished loading obs, the shape is" + str(obs_h.shape) + ".")
    reward_h = th.tensor(np.array(hdFile_r.get('reward')))
    print("Finished loading reward, the shape is" + str(reward_h.shape) + ".")
    state_h = th.tensor(np.array(hdFile_r.get('state')))
    print("Finished loading state, the shape is" + str(state_h.shape) + ".")
    terminated_h = th.tensor(np.array(hdFile_r.get('terminated')))
    print("Finished loading terminated, the shape is" + str(terminated_h.shape) + ".")
    return actions_h, actions_onehot_h, avail_actions_h, filled_h, obs_h, reward_h, state_h, terminated_h