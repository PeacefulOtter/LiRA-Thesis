

from src.types import WayIRI, Dataset

import numpy as np
import pandas as pd
import requests
import json 

def req_valhalla_service(chunk):
    url = 'https://valhalla1.openstreetmap.de/trace_attributes' 
    df = pd.DataFrame(data={
        'lon': [p.lon for p in chunk], 
        'lat': [p.lat for p in chunk], 
        # 'time': [p.ts for p in chunk]
    })
    coords = df.to_json(orient='records')
    data = '{"shape":' + str(coords) + ""","search_radius": 0, "shape_match":"map_snap", "costing":"auto", "format":"osrm"}"""
    res = requests.post(url, data=data, headers={'Content-type': 'application/json'})
    return json.loads(res.text)

def map_match(data: np.ndarray):

    chunk = req_valhalla_service(data)

    if 'code' in chunk:
        print(json.dumps(chunk, indent=2))

    elif 'error_code' in chunk:
        print(json.dumps(chunk, indent=2))
        return [], []

    if 'code' in chunk and chunk['code'] == "DistanceExceeded":
        print('DistanceExceeded, splitting data into 2 and retrying')
        datas = np.array_split(data, 2)
        mm_points_1, way_ids_1 = map_match(datas[0])
        mm_points_2, way_ids_2 = map_match(datas[1])
        return mm_points_1 + mm_points_2, way_ids_1 + way_ids_2

    if 'edges' not in chunk:
        print('edges not in chunk, returning')
        return [], []


    # get way_ids and edge_lengths (in meters)
    way_ids = [edge['way_id'] for edge in chunk['edges']]
    edge_lengths = [int(edge['length'] * 1000.0) for edge in chunk['edges']]

    from itertools import groupby
    from operator import itemgetter
    from functools import reduce

    way_lengths = {}
    for way_id, group in groupby(zip(way_ids, edge_lengths), key=itemgetter(0)):
        way_lengths[way_id] = reduce(lambda accumulator, element: accumulator + int(element[1]), group, 0)
    
    ways_data = {}
    prev_dist = 0
    prev_way_id = 0
    prev_edge_index = 0
    prev_way_dist = 0

    for i, pos in enumerate(chunk['matched_points']):
        edge_index = pos['edge_index'] 
        dist = pos['distance_along_edge']

        # find out the edge_index in case Valhalla does not provide it
        if edge_index >= len(way_ids):
            if prev_dist > dist:
                edge_index = min( prev_edge_index + 1, len(way_ids) - 1 )
            else:
                edge_index = prev_edge_index

        way_id = way_ids[edge_index]
        edge_length = edge_lengths[edge_index]

        # calculate previous way distance
        if edge_index != prev_edge_index:
            prev_edge_length = edge_lengths[prev_edge_index]

            if way_id == prev_way_id:
                prev_way_dist += prev_edge_length
            else:
                prev_way_dist = 0

        edge_dist = dist * edge_length
        way_dist = prev_way_dist + edge_dist

        if way_id not in ways_data: 
            ways_data[way_id] = Dataset()

        ways_data[way_id].append(way_dist, data[i].value)

        prev_dist = dist
        prev_way_id = way_id
        prev_edge_index = edge_index

    
    return ways_data, way_lengths
