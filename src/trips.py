

import numpy as np
import json

from src.types import LatLonIRI, Dataset
from src.map_match import map_match
from src.database import query_postgres

def parseIRI(row):
    obj = json.loads(row['message'])

    if obj['IRI5'] == None and obj['IRI21'] == None:
        return None

    return LatLonIRI(
        row['lat'], 
        row['lon'], 
        (obj['IRI5'] + obj['IRI21']) / 2, 
        str(row['Created_Date'])
    )


def query_trip(trip_id):
    query = f'''
    SELECT "lat", "lon", "message", "Created_Date" FROM "DRDMeasurements" 
    WHERE ("FK_Trip" = \'{trip_id}\' and "T"=\'IRI\' and 
    "lat" IS NOT NULL and "lon" IS NOT NULL and "lat" != 0 and "lon" != 0)
    ORDER BY cast("TS_or_Distance" as integer)
    ''' 
    res = query_postgres(query)
    res = res.apply( parseIRI, axis=1 )
    res = np.array(res.tolist())
    res = res[res != None]
    return map_match(res)


def query_trips(trip_ids):

    ways_datas = {}
    
    # query trips and store regroup them per way_id
    for trip_id in trip_ids: 
        ways_data, way_lengths = query_trip(trip_id)
        for way_id, way_data in ways_data.items():
            if way_id not in ways_datas:
                ways_datas[way_id] = Dataset()
            ways_datas[way_id].append(way_data.X, way_data.y)

    print(ways_datas)

    # sort the IRIs in each way per way_dist
    for way_id, data in ways_datas.items():
        data.sort()

    return ways_datas, way_lengths