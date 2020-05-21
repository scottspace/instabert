#
## Google Cloud Function /city
# 
# /city?lat=xx.x&lng=yy.y
#
# Returns a json array of metadata for the closest cities
# for a point at latitude xx.x and longitude yy.y
# where xx.y and yy.y are in degrees.
#

from google.cloud import bigquery
import pandas as pd 
import flask

# a template to feed a longitude,latitude pair
tpl = """
WITH a AS (
  # a table with points around the world
  SELECT * FROM UNNEST([ST_GEOGPOINT({}, {})]) my_point
), b AS (
  # any table with cities world locations
  SELECT *, ST_GEOGPOINT(lon,lat) latlon_geo
  FROM `fh-bigquery.geocode.201806_geolite2_latlon_redux` 
)
SELECT my_point, city_name, subdivision_1_name, country_name, continent_name
FROM (
  SELECT loc.*, my_point
  FROM (
    SELECT ST_ASTEXT(my_point) my_point, ANY_VALUE(my_point) geop
      , ARRAY_AGG( # get the closest city
           STRUCT(city_name, subdivision_1_name, country_name, continent_name) 
           ORDER BY ST_DISTANCE(my_point, b.latlon_geo) LIMIT 1
        )[SAFE_OFFSET(0)] loc
    FROM a, b 
    WHERE ST_DWITHIN(my_point, b.latlon_geo, 100000)  # filter to only close cities
    GROUP BY my_point
  )
)
GROUP BY 1,2,3,4,5
"""

def hello_world(request):
    """Responds to any HTTP request.
    Args:
        request (flask.Request): HTTP request object.
    Returns:
        The response text or any set of values that can be turned into a
        Response object using
        `make_response <http://flask.pocoo.org/docs/1.0/api/#flask.Flask.make_response>`.
    """
    lat = 0.0
    lng = 0.0
    if request.args:
        if 'lat' in request.args:
            lat = float(request.args.get('lat'))
        if 'lng' in request.args:
            lng = float(request.args.get('lng'))
    response = flask.jsonify(get_city(lat,lng))
    response.headers.set('Access-Control-Allow-Origin', '*')
    response.headers.set('Access-Control-Allow-Methods', 'GET, POST')
    return response

def get_city(lat,lng):
   # BQ Query to lookup closest city
   global tpl
   QUERY = tpl.format(lng,lat)
   bq_client = bigquery.Client()
   query_job = bq_client.query(QUERY) # API request
   rows_df = query_job.result().to_dataframe() # Waits for query to finish
   return rows_df.to_dict(orient='records')