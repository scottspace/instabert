#
## Google Cloud function  /articles
# 
#
# articles/?t=n    fetch articles from topic N
# articles/?w=name fetch articles written by 'name'
# articles         fetch all articles
#  
# We return a JSON array of article descriptions
#

from google.cloud import bigquery
import pandas as pd 
import flask

def hello_world(request):
    """Responds to any HTTP request.
    Args:
        request (flask.Request): HTTP request object.
    Returns:
        The response text or any set of values that can be turned into a
        Response object using
        `make_response <http://flask.pocoo.org/docs/1.0/api/#flask.Flask.make_response>`.
    """
    request_json = request.get_json()
    if request.args and 't' in request.args:
        response = flask.jsonify(get_topic(request.args.get('t')))
    elif request.args and 'w' in request.args:
        response = flask.jsonify(get_who(request.args.get('w')))
    else:
        response = flask.jsonify(get_articles())
    response.headers.set('Access-Control-Allow-Origin', '*')
    response.headers.set('Access-Control-Allow-Methods', 'GET, POST')
    return response

def get_topic(t):
   # BQ Query to get add to cart sessions
   QUERY = "SELECT a.*,b.name as topic_text from `octo-news.gdelt_sa.daily_feed` as a inner join "
   QUERY += "`octo-news.gdelt_sa.themes` as b on a.topic=b.index where a.topic="+str(t)+" order by index limit 50"
   bq_client = bigquery.Client()
   query_job = bq_client.query(QUERY) # API request
   rows_df = query_job.result().to_dataframe() # Waits for query to finish
   return rows_df.to_dict(orient='records')

def get_who(w):
   # BQ Query to get add to cart sessions
   QUERY = "SELECT a.*,b.name as topic_text from `octo-news.gdelt_sa.daily_feed` as a inner join "
   QUERY += "`octo-news.gdelt_sa.themes` as b on a.topic=b.index where "
   QUERY += " a.who like '"+str(w)+"' order by index limit 50"
   bq_client = bigquery.Client()
   query_job = bq_client.query(QUERY) # API request
   rows_df = query_job.result().to_dataframe() # Waits for query to finish
   return rows_df.to_dict(orient='records')

def get_articles():
   # BQ Query to get add to cart sessions
   QUERY = "SELECT a.*,b.name as topic_text from `octo-news.gdelt_sa.daily_feed` as a inner join "
   QUERY += "`octo-news.gdelt_sa.themes` as b on a.topic=b.index order by index limit 50"
   bq_client = bigquery.Client()
   query_job = bq_client.query(QUERY) # API request
   rows_df = query_job.result().to_dataframe() # Waits for query to finish
   return rows_df.to_dict(orient='records')
