import numpy as np
import torch
import warnings

'''
Hierarchy:
                 reducer_group
               /       |       \
              /        |        \
    reducer_1     reducer_2   reducer_3 ...
'''

class reducer:
    """Bind one blob to a reducer for collection of batch data."""
    def __init__(self, post_func=None, as_numpy=True, **func_kwargs):
        self.batch_data_list = []  # for collecting batch data.
        self.nr_batch = 0  # number of batch collected.
        self.blob_handler = (post_func, func_kwargs)
        self.as_numpy     = as_numpy

    def squeeze(self, arr, dim=None):
        if isinstance(arr, np.ndarray):
            return np.squeeze(arr, axis=dim)
        elif isinstance(arr, torch.Tensor):
            # Although dim is optional args in pytorch, it's not allowed to be None when it's explicitly specified.
            return torch.squeeze(arr, dim=dim)  if (dim is not None)  else torch.squeeze(arr)
        else:
            raise NotImplementedError(f"Unknown array type: {type(arr)}")

    def cat(self, list_arr, dim=0):
        assert isinstance(list_arr, list) or isinstance(list_arr, tuple), type(list_arr)
        _arr = list_arr[0]
        if   isinstance(_arr, np.ndarray):
            return np.concatenate(list_arr, axis=dim)
        elif isinstance(_arr, torch.Tensor):
            return torch.cat(list_arr, dim=dim )
        else:
            raise NotImplementedError(f"Unknown array type: {type(_arr)}")

    def reset(self):
        self.batch_data_list = []  # for collecting batch data.
        self.nr_batch = 0  # number of batch collected.

    def resume(self, pre_batch_data):
        self.batch_data_list = [pre_batch_data]

    def collect(self, batch_data, squeeze=True, detach=True):
        post_func, func_kwargs = self.blob_handler

        is_arr = isinstance(batch_data, (np.ndarray, torch.Tensor))
        if not is_arr: # batch_data is a scalar
            batch_data = np.array(batch_data)
        #
        if isinstance(batch_data, torch.Tensor):
            if self.as_numpy:
                batch_data = batch_data.data.cpu().numpy().copy()  # as numpy array
            else:
                batch_data = batch_data.data
                # 'To detach computational graph here.'
                if detach: # it is always wise to call detach to avoid unwanted behavior.
                    batch_data = batch_data.detach()     # .cpu().numpy().copy()

        if squeeze:
            batch_data = self.squeeze(batch_data) #.reshape((-1,))

        if post_func is not None:
            batch_data = post_func( batch_data, **func_kwargs )
            if squeeze:
                batch_data = self.squeeze(batch_data)

        if batch_data.shape==():
            # TODO:  what if reduce array is not of shape (batch_size, 1), but (batch_size, c, h, w)?
            batch_data = batch_data.reshape((-1,)) # hack for preventing squeeze single value array.



        self.batch_data_list.append(batch_data)
        self.nr_batch += 1

        # just return a copy of batch_data in case needed
        return batch_data

    def reduce(self, reset=False): #, blobs=None):
        assert len(self.batch_data_list)>0, "[Exception] No data to reduce."
        concated_data = self.cat(self.batch_data_list, dim=0)
        if reset:
            self.reset()
        if isinstance(concated_data, torch.Tensor) and self.as_numpy:
            warnings.warn("this should not happen here because in self.collect torch.Tensor has been casted to numpy.")  # ---------------------------> To delete
            if concated_data.is_cuda:
                concated_data = concated_data.data.cpu()
            return concated_data.numpy().copy()
        else:
            return concated_data

class reducer_group:
    def __init__(self, target_names, post_func=None, as_numpy=True, **func_kwargs):
        self.names = target_names  # name is gt, pred, scr, loss
        self.name2reducer  = {}
        for name in self.names:
            self.name2reducer[name] = reducer(post_func=post_func, as_numpy=as_numpy, **func_kwargs)

    def reset(self):
        for name in self.names:
            self.name2reducer[name].reset()

    def resume(self, name2pre_batch_data):
        for name in self.names:
            self.name2reducer[name].resume( name2pre_batch_data[name] )

    def collect(self, tgts_dict, squeeze=True, detach=True):
        """collect add new batch data to list."""
        name2batch_data = {}
        # for name, var in tgts_dict.items():
        for name in self.names:
            var = tgts_dict[name]
            batch_data = self.name2reducer[name].collect(var, squeeze=squeeze, detach=detach)
            name2batch_data[name] = batch_data
        # just return a copy of batch_data in case needed
        return name2batch_data

    def reduce(self, reset=False):
        """reduce only return data, will change anything."""
        name2data = {} # edict()
        for name, reducer in self.name2reducer.items():
            name2data[name] = reducer.reduce()
        if reset:
            self.reset()
        return name2data



if __name__ == '__main__':
    rd = reducer_group(['step', 'value'])
    for i in range(10):
        rd.collect(dict(step=i, value=i**2))
    print(rd.reduce())
