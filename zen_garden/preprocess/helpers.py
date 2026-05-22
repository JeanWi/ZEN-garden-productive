import json
from pathlib import Path

import numpy as np
import pandas as pd
import os


def read_attribute_json(folder_path, filename="attributes"):
    """Loads raw attribute json file for an element

    :param filename: name of attributes file, default is 'attributes'
    :return: attribute_dict.
    """
    file_path = Path(folder_path) / f"{filename}.json"

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Attributes file does not exist for {file_path}"
        )

    with open(file_path, "r") as file:
        data = json.load(file)
    attribute_dict = {}
    for k, v in data.items():
        if isinstance(v, list):
            attribute_dict[k] = {sk: sv for d in v for sk, sv in d.items()}
        else:
            attribute_dict[k] = v

    return attribute_dict


def read_input_csv(folder_path, input_file_name):
    """Reads raw input data for a parameter and returns it as a dataframe.

    :param input_file_name: name of selected file
    :return: df_input: pd.DataFrame with input data
    """
    # append .csv suffix
    file_path = Path(folder_path) / f"{input_file_name}.csv"
    if not file_path.exists():
        return None

    df_input = pd.read_csv(
        file_path,
        header=0,
        index_col=None,
)
    # check for header name duplicates (pd.read_csv() adds a dot and a
    # number to duplicate headers)
    if any("." in col for col in df_input.columns):
        raise AssertionError(
            f"The input data file {input_file_name} at "
            f"{folder_path} contains two identical header names."
        )
    return df_input

def read_used_nodes(folder_path, set_nodes_config):
    df_nodes_w_coords = read_input_csv(folder_path, "set_nodes")
    set_nodes_input = df_nodes_w_coords["node"].to_list()

    # if no nodes specified in system, use all nodes
    if len(set_nodes_config) == 0 and not len(set_nodes_input) == 0:
        set_nodes_config = set_nodes_input
    else:
        missing_nodes = list(
            set(set_nodes_config).difference(set_nodes_input)
        )
        assert len(missing_nodes) == 0, (
            f"The nodes {missing_nodes} were declared in the "
            "config but do not exist in the input file "
            f"{os.path.join(folder_path, 'set_nodes')}"
        )
    set_nodes_config.sort()
    return set_nodes_config

def read_coordinates_of_used_nodes(folder_path, set_nodes):
    df_nodes_w_coords = read_input_csv(folder_path, "set_nodes")
    if len(set_nodes) != 0:
        df_nodes_w_coords = df_nodes_w_coords[
            df_nodes_w_coords["node"].isin(set_nodes)
        ]
    return df_nodes_w_coords


def read_edges(folder_path, set_nodes, consistency_check):
    set_edges_input = read_input_csv(folder_path, "set_edges")
    consistency_check(set_edges_input=set_edges_input)
    if set_edges_input is not None:
        set_edges = set_edges_input[
            (set_edges_input["node_from"].isin(set_nodes))
            & (set_edges_input["node_to"].isin(set_nodes))
            ]
        set_edges = set_edges.set_index("edge")
        return set_edges
    else:
        raise FileNotFoundError(
            f"Input file set_edges.csv is missing from {folder_path}"
        )

def single_node_systems_check(set_nodes, set_transport_technologies):
    assert (
            len(set_nodes) > 1
            or len(set_transport_technologies) == 0
    ), (
        f"Only one node is given in the system file. "
        f"Transport technologies are not allowed in this case. "
        f"You selected {set_transport_technologies}"
    )

def calculate_haversine_distances_from_nodes(nodes_with_locations, set_nodes_on_edges, unit_handling):
    """Computes the distance (in km) between two nodes.

    The Haversine function is used to compute the distance in kilometers based on
     their lon lat coordinates.

    :return: dict containing all edges along with their distances
    """
    set_haversine_distances_of_edges = {}

    # convert coords from decimal degrees to radians
    nodes_with_locations["lon"] = nodes_with_locations["lon"] * np.pi / 180
    nodes_with_locations["lat"] = nodes_with_locations["lat"] * np.pi / 180
    # Radius of the Earth in kilometers
    radius = 6371.0
    for edge, nodes in set_nodes_on_edges.items():
        node_1, node_2 = nodes
        coords1 = nodes_with_locations[nodes_with_locations["node"] == node_1]
        coords2 = nodes_with_locations[nodes_with_locations["node"] == node_2]
        # Haversine formula
        dlon = coords2["lon"].squeeze() - coords1["lon"].squeeze()
        dlat = coords2["lat"].squeeze() - coords1["lat"].squeeze()
        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(coords1["lat"].squeeze())
            * np.cos(coords2["lat"].squeeze())
            * np.sin(dlon / 2) ** 2
        )
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        distance = radius * c
        set_haversine_distances_of_edges[edge] = distance
    multiplier = unit_handling.get_unit_multiplier(
        "km", attribute_name="distance"
    )
    set_haversine_distances_of_edges = {
        key: value * multiplier
        for key, value in set_haversine_distances_of_edges.items()
    }
    return set_haversine_distances_of_edges