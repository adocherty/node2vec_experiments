import os
import pandas as pd
import numpy as np
from collections import Counter
from numba import jit

@jit(nopython=True, parallel=True)
def remap_ids(data, uid_map, mid_map):
    """
    Remap user and movie IDs
    """
    Nm = mid_map.shape[0]
    for ii in range(data.shape[0]):
        mid, uid = data[ii, :]
        new_mid = np.searchsorted(mid_map, mid)
        new_uid = np.searchsorted(uid_map, uid)

        # Only map to index if found, else map to zero
        data[ii,1] = new_uid + Nm if uid_map[new_uid] == uid else -1
        data[ii,0] = new_mid


def preprocess_nf(handler, inputs, output, parameters):
    print(inputs)
    input_spec = inputs['in_data']
    edgelist_name = input_spec['location']
    dataset_name = input_spec['dataset_name']

    # Load edgelist
    data = pd.read_csv(edgelist_name, names=['mId', 'uId'],
                       sep=" ", header=None, dtype='int32')

    # Enumerate movies
    mids = np.unique(data['mId'])

    # Number of movies rated by each user
    movies_per_user = Counter(data['uId'])

    # Filter users by count and enumerate
    uids_all = np.unique(data['uId'])
    uids = np.array([uId for uId in uids_all if movies_per_user[uId] > 1])

    # Filter data and transform
    remap_ids(data.values, uids, mids)

    # Remove users with invalid ID
    data = data.query('uId>=0')

    # Node ids
    Nm = len(mids);
    Nu = len(uids)
    movie_id_map = {i: "m_{}".format(mId) for i, mId in enumerate(mids)}
    user_id_map = {i + Nm: "u_{}".format(uId) for i, uId in enumerate(uids)}
    id_map = {**movie_id_map, **user_id_map}
    inv_id_map = dict(zip(id_map.values(), id_map.keys()))

    # ID maps for node2vec
    default_map = {i: i for i in range(Nm + Nu)}

    # Save as graph in required formats
    edge_list_filename = os.path.join(output, "nf_edgelist_remap.txt")
    data.to_csv(edge_list_filename, sep=' ', header=False, index=False)

    vertex_map_loc = handler.save_as_pickle(id_map, "vertex_map.pkl")
    inv_vertex_map_loc = handler.save_as_pickle(inv_id_map, "inv_vertex_map.pkl")
    # unique_labels_loc = worker.save_as_pickle(unique_labels, "unique_labels.pkl")

    task_results = {
        'vertex_map_filename': vertex_map_loc,
        'inv_vertex_map_filename': inv_vertex_map_loc,
        'unique_labels_filename': None,
        'features_filename': None,
        'vertex_labels_filename': None,
        'graph_filename': edge_list_filename,
        'dataset_name': dataset_name,
    }
    return task_results
