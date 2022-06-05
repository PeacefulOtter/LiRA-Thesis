

import numpy as np
import json

from src.types import LatLonIRI
from src.map_match import map_match
from src.database import query_postgres

def parseIRI(row):
    obj = json.loads(row['message'])

    if obj['IRI5'] == None and obj['IRI21'] == None:
        return None

    iri = (obj['IRI5'] + obj['IRI21']) / 2

    return LatLonIRI(
        row['lat'], 
        row['lon'], 
        iri, 
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

    datas = {}
    
    # query trips and store them per way_id 
    for trip_id in trip_ids: 
        for pos in query_trip(trip_id):
            if pos.way_id not in datas:
                datas[pos.way_id] = np.array([])
            datas[pos.way_id] = np.append(datas[pos.way_id], pos)

    # sort the IRIs in each way per way_dist
    for way_id, iris in datas.items():
        datas[way_id] = sorted(iris, key=lambda x: x.way_dist) 

    return datas