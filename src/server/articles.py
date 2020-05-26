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
    if request.args and 't' in request.args:
        response = flask.jsonify(get_topic(request.args.get('t')))
    elif request.args and 'w' in request.args:
        response = flask.jsonify(get_who(request.args.get('w')))
    elif request.args and 'm' in request.args:
        response = flask.jsonify(get_map(request.args.get('m')))
    elif request.args and 'c' in request.args:
        response = flask.jsonify(get_articles(request.args.get('c')))
    else:
        response = flask.jsonify(get_articles())
    response.headers.set('Access-Control-Allow-Origin', '*')
    response.headers.set('Access-Control-Allow-Methods', 'GET, POST')
    return response

def with_a(city = ''):
    q = """
with a as 
(SELECT *
FROM
(
    SELECT *, ROW_NUMBER() OVER (PARTITION BY index ORDER BY date  desc) rn
    FROM `octo-news.gdelt_sa.daily_map_feed` 
    where lower(city) like lower('%$CITY%')
) t
WHERE rn = 1
order by index)
"""
    return q.replace('$CITY',city)


def get_topic(t):
   # BQ Query to get articles with topic t
   QUERY = with_a()
   QUERY += "SELECT a.*,b.name as topic_text from a inner join "
   QUERY += "`octo-news.gdelt_sa.themes` as b on a.topic=b.index "
   QUERY += "where a.topic="+str(t)+" order by index limit 50"
   bq_client = bigquery.Client()
   query_job = bq_client.query(QUERY) # API request
   rows_df = query_job.result().to_dataframe() # Waits for query to finish
   return rows_df.to_dict(orient='records')

def get_who(w):
   # BQ Query to get articles authored by w
   QUERY = with_a()
   QUERY += "SELECT a.*,b.name as topic_text from a inner join "
   QUERY += "`octo-news.gdelt_sa.themes` as b on a.topic=b.index where "
   QUERY += " a.who like '"+str(w)+"' order by index limit 50"
   bq_client = bigquery.Client()
   query_job = bq_client.query(QUERY) # API request
   rows_df = query_job.result().to_dataframe() # Waits for query to finish
   return rows_df.to_dict(orient='records')

def get_articles(city=''):
   # BQ Query to get top 50 articles
   QUERY = with_a(city)
   QUERY += "SELECT a.*,b.name as topic_text from a inner join "
   QUERY += "`octo-news.gdelt_sa.themes` as b on a.topic=b.index order by index limit 50"
   bq_client = bigquery.Client()
   query_job = bq_client.query(QUERY) # API request
   rows_df = query_job.result().to_dataframe() # Waits for query to finish
   return rows_df.to_dict(orient='records')

def get_map(m):
   # BQ Query to get top 250 articles in city m, 'all' for all
   if m.lower() == 'all':
       m = ''
   QUERY = with_a(m)
   QUERY += "SELECT a.*,b.name as topic_text from a inner join "
   QUERY += "`octo-news.gdelt_sa.themes` as b on a.topic=b.index "
   QUERY += "order by index limit 250"
   bq_client = bigquery.Client()
   query_job = bq_client.query(QUERY) # API request
   rows_df = query_job.result().to_dataframe() # Waits for query to finish
   return rows_df.to_dict(orient='records')
