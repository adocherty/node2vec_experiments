import os
import pandas as pd
import numpy as np
from collections import Counter
from numba import jit
import networkx as nx

@jit(nopython=True, parallel=True)
def remap_ids(data, uid_map, mid_map, uid_inx=0, mid_inx=1):
    """
    Remap user and movie IDs
    """
    Nm = mid_map.shape[0]
    Nu = uid_map.shape[0]
    for ii in range(data.shape[0]):
        mid =  data[ii, mid_inx]
        uid = data[ii, uid_inx]

        new_mid = np.searchsorted(mid_map, mid)
        new_uid = np.searchsorted(uid_map, uid)

        if new_mid < 0:
            print(mid, new_mid)

        # Only map to index if found, else map to zero
        if new_uid < Nu and (uid_map[new_uid] == uid):
            data[ii, uid_inx] = new_uid + Nm
        else:
            data[ii, uid_inx] = -1
        data[ii, mid_inx] = new_mid


def preprocess_nf(handler, inputs, output, parameters):
    input_spec = inputs['in_data']
    edgelist_name = input_spec['location']
    dataset_name = parameters['dataset_name']

    # Columns should contain mId, uId & score
    columns = parameters.get("columns", ['mId', 'uId', 'score'])
    usecols = parameters.get("usecols", [0,1,3])
    sep = parameters.get("sep", ",")

    # Load edgelist
    data = pd.read_csv(edgelist_name,
                       names=columns,
                       sep=sep,
                       header=None,
                       usecols=usecols,
                       dtype='int64')

    # Enumerate movies & users
    mids = np.unique(data['mId'])
    uids = np.unique(data['uId'])
    Nm = len(mids)
    Nu = len(uids)

    # Number of movies rated by each user
    reviews_per_user = Counter(data['uId'])

    # Filter data and transform
    remap_ids(data.values, uids, mids)

    # Remove users with invalid ID
    #data = data.query('uId>=0')

    # Node ID map back to movie and user IDs
    movie_id_map = {i: "m_{}".format(mId) for i, mId in enumerate(mids)}
    user_id_map = {i + Nm: "u_{}".format(uId) for i, uId in enumerate(uids)}
    id_map = {**movie_id_map, **user_id_map}
    inv_id_map = dict(zip(id_map.values(), id_map.keys()))

    # Save as graph in required formats
    edge_list_filename = os.path.join(
        output, "{}_edge_homogeneous.txt".format(dataset_name))
    data.to_csv(edge_list_filename, sep=' ',
                columns=['uId', 'mId'], header=False, index=False)

    edge_label_filename = os.path.join(
        output, "{}_edge_labels.txt".format(dataset_name))
    data.to_csv(edge_label_filename, sep=' ',
                columns=['score'], header=False, index=False)

    # Node2Vec uses the weight edge attribute
    data['weight'] = 1

    # Create split information
    # Currently this is hard coded to repllicate the movielens split 1
    # Later we should split this out
    data['split'] = 0
    data['split'].iloc[:20000] = 1

    # Create networkx graph
    G = nx.from_pandas_edgelist(data, source='uId', target='mId', edge_attr=True)
    G_loc = handler.save_as_pickle(
        G, "{}_graphnx.pkl".format(dataset_name))

    vertex_map_loc = handler.save_as_pickle(
        id_map, "{}_vertex_map.pkl".format(dataset_name))
    inv_vertex_map_loc = handler.save_as_pickle(
        inv_id_map, "{}_inv_vertex_map.pkl".format(dataset_name))

    task_results = {
        'vertex_map_filename': vertex_map_loc,
        'inv_vertex_map_filename': inv_vertex_map_loc,
        'graph_filename': edge_list_filename,
        'labels_filename': edge_label_filename,
        'nx_filename': G_loc,
        'dataset_name': dataset_name,
    }
    return task_results
