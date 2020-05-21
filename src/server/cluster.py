# !pip -q install nltk requests
#
# TODO please clean me up, OO me. This is raw notebook sludge.
#
import requests
import os
import random
import numpy as np
import pandas as pd
import nltk
import json
import re
from tqdm import tqdm
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize
from google.cloud import bigquery
from sklearn.decomposition import LatentDirichletAllocation
from collections import Counter
from sklearn import metrics
import keras
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

if len(sys.argv) > 1:
    a = sys.argv
    a.pop(0) # skip command name
    city_name = ' '.join(a)
else:
    city_name = 'San Francisco'
print("Creating news cluster for "+city_name)

# hyperparameters
project_id = 'octo-news'
no_topics = 32  # number of sentence topics of LDA
C = 16 # Number of sentence clusters of [LDA,Bert]
no_features = 256

# Bert
BERT_HTTP = 'http://bert.scott.ai/encode'
BERT_TENSOR_SIZE = 768                                              
BERT_TENSOR_DISTANCE = "angular"
BERT_TENSOR_DB = "bert.db"
BERT_API_COUNT = 0

#---------------
print("Downloading data for "+city_name)
# get our sentence tokenizer                                                                                          
nltk.download('punkt')
client = bigquery.Client(project=project_id)
query = """
SELECT   distinct(url_orig) as url,
         keyimage as image,
         domain_root as domain,
         page_title as title, 
         score, 
         date,
         z,
         page_ftxt as text 
FROM
(
    SELECT *, ROW_NUMBER() OVER (PARTITION BY keyimage ORDER BY date desc) rn
    FROM `octo-news.gdelt_sa.daily_reputable_refresh`
    where length(keyimage) >  0 and lower(city) like '%$CITY%'
    limit 50000
) t
WHERE rn = 1
"""
q = client.query(query.replace('$CITY',city_name.lower()))
for row in q:
  print("url={} title={}".format(row['url'], row['title']))
  break

data_df = q.to_dataframe()  # Download all

#setup some constants to configure the Bert service

data = []
db = {'doc_id': [],
      'url': [],
      'z': [],
      'city': [],
      'title': [],
      'image_url': [],
      'author': [],
      'domain': [],
      'date': [],
      'vindex': []}
vecdb = {'sid': [], 'doc_id': [], 'text': [], 'tensor': []}

# Simple BERT utilities   

def bertify_array(text_array):
    "Turn an array of text, text_array, into an array of tensors. Sentences are best."
    global BERT_API_COUNT
    # eid is our encoding id, which we really don't use as                                                            
    # bert is synchronous over http.                                                                                  
    r = requests.post(BERT_HTTP, 
                      json={"id": BERT_API_COUNT, 
                            "texts": text_array, 
                            "is_tokenized": False})
    v = r.json()
    BERT_API_COUNT += 1
    try:
        if (v['status'] == 200):
            return np.array(v['result'])
        else:
          print("Unexpected Bert status: ",v['status'])
    except:
        print("Unexpected Bert error: ",sys.exc_info()[0])
    return None

def bertify(text):
    "Turn text into a tensor, sentences are best."
    ans = bertify_array([text])
    if ans is not None:
        ans = ans[0]
    return ans

# make sure our images are clean
def valid_image(url):
  "Return True if url is a valid image"
  r = requests.get(url)
  if r.status_code == 200:
    kind = r.headers.get('Content-Type','')
    return kind.lower().startswith('image')
  return False

def insert_db_entry(entry):
  global city_name
  n = len(db['doc_id']) 
  info = {'doc_id': n,
          'url': entry['url'],
          'title': entry['title'],
          'image_url': entry['image'],
          'city': city_name,
          'author':'',
          'z': entry['z'],
          'domain': entry['domain'],
          'date': entry['date'],
          'vindex': len(vecdb)}
  for key in info:
    db[key].append(info[key])
  return n

def insert_vecdb_entries(doc_id, sents, vecs):
    for i in range(0,len(vecs)):
     # get our sentence id
      sid = len(vecdb['sid'])
      # record our sentence and point to the db entry
      # that tells us more about the article from which
      # it came
      vecdb['sid'].append(sid)
      vecdb['doc_id'].append(doc_id)
      vecdb['text'].append(sents[i])
      vecdb['tensor'].append(vecs[i])

def process_entry(entry): 
  if (not valid_image(entry['image'])):
    return None
  sents = first_clean_sentences(entry['text'])
  clean_title_sent = sentences(entry['title'])
  if (len(clean_title_sent) > 0):
    # we've seen blank titles, and entire docs as titles
    clean_title = clean_title_sent[0]
    if (len(clean_title) > 0):
      sents.insert(0,clean_title) 
  vecs = bertify_array(sents)
  if vecs is None:
    return None
  n  =  insert_db_entry(entry)
  insert_vecdb_entries(n, sents, vecs)
  return sents

def first_clean_sentences(text, k=50):
  sent = sentences(text)
  valid = []
  # k clean sentences                                                                                                 
  while len(sent) > 0 and k > 0:
    s = sent.pop(0)
    if s.find('EOP') < 0  and len(s) > 10:
      valid.append(s)
      k = k-1
  return valid

def sentences(text):
  text = clean_text(text)
  return sent_tokenize(text)

def clean_text(text):
  r1  = re.compile(r' (\w+)\.(\w+) ')
  r2 = re.compile(r' - ')
  text = text.replace("\n\n"," EOP ")
  #text = text.replace(".",". ")
  text = text.replace("\t"," ")
  text = text.replace("\n"," ")
  text = remove_html_tags(text)
  text = re.sub(r1,r' \1. \2 ',text,99)
  text = re.sub(r2,'. ',text,99)
  return text

def remove_html_tags(text):
    """Remove html tags from a string"""
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def showv(v):
  n = vecdb['doc_id'][v]
  sentence  = vecdb['text'][v]
  print("  "+db['domain'][n]+": "+db['title'][n])
  print("  \""+sentence+"\"")
  print("  "+db['url'][n])

#---------------
# load sample data
print("Loading data into memory...\n")
data=[]
from urllib.parse import urlparse
for index, row in data_df.iterrows():
  who = urlparse(row['url']).netloc
  if index < 10:
    print(who,':',row['title'])
  data.append(row)
print("Loaded",len(data),"items.")

# process the sample data of ~1000 articles (5 min)

def bert_do(n=1000):
  count = 0
  for i in tqdm(range(0,min(n*2,len(data)))):
    #print(data[i]['page_title'])
    if process_entry(data[i]) is not None:
      count += 1
      if (count == n):
        break

#---------------
print("Pulling the most recent 2000 articles with valid images")
bert_do(2000)

# From github.com/scottspace/contextual_topic_identification

class Autoencoder:
    """
    Simple autoencoder for learning latent space representation
    architecture simplified for only one hidden layer
    """

    def __init__(self, latent_dim=32, activation='relu', epochs=200, batch_size=128):
        self.latent_dim = latent_dim
        self.activation = activation
        self.epochs = epochs
        self.batch_size = batch_size
        self.autoencoder = None
        self.encoder = None
        self.decoder = None
        self.his = None

    def _compile(self, input_dim):
        """
        compile the computational graph
        """
        input_vec = Input(shape=(input_dim,))
        encoded = Dense(self.latent_dim, activation=self.activation)(input_vec)
        decoded = Dense(input_dim, activation=self.activation)(encoded)
        self.autoencoder = Model(input_vec, decoded)
        self.encoder = Model(input_vec, encoded)
        encoded_input = Input(shape=(self.latent_dim,))
        decoder_layer = self.autoencoder.layers[-1]
        self.decoder = Model(encoded_input, self.autoencoder.layers[-1](encoded_input))
        self.autoencoder.compile(optimizer='adam', loss=keras.losses.mean_squared_error)

    def fit(self, X, verbose=0):
        if not self.autoencoder:
            self._compile(X.shape[1])
        X_train, X_test = train_test_split(X)
        self.his = self.autoencoder.fit(X_train, X_train,
                                        epochs=200,
                                        batch_size=128,
                                        shuffle=True,
                                        validation_data=(X_test, X_test), 
                                        verbose=verbose)

# create dataframes for testing

#---------------
print("Setting up for analysis with Pandas")
doc_df = pd.DataFrame(data=db)
sent_df = pd.DataFrame(data=vecdb)

#---------------
print("Creating document corpus")
nDocs = np.max(doc_df['doc_id'])
DocText = [" "]*nDocs
for i in range(nDocs):
  text_i = "\n ".join(sent_df[sent_df.doc_id == i]['text'].values.flatten())
  DocText[i] = text_i

#---------------
print("Creating sentence corpus")
nSents = np.max(sent_df['sid'])
SentText = [" "]*nSents
for i in range(nSents):
  text_i = "\n ".join(sent_df[sent_df.sid == i]['text'].values.flatten())
  SentText[i] = text_i

## Create word frequency
## We want words that are in at least 3 articles,
## but no more than 25% of the corpus.
##
 
#---------------
print("Counting word frequencies")
corpus = DocText
#corpus = SentText
vectorizer = CountVectorizer(min_df=3, max_df=0.25)
DocX = vectorizer.fit_transform(corpus)
print("Found",len(vectorizer.get_feature_names()),"interesting document words")

#---------------
print("Calculating TF/IDF values")
tfv = TfidfVectorizer(min_df=3, max_df=0.25, stop_words='english')
doc_tf = tfv.fit_transform(corpus)

def describe_doc(d_i):
  df = doc_df[doc_df.doc_id == d_i]
  print(df['title'].values[0])
  print(df['url'].values[0])
  top = np.argsort(doc_tf[d_i].toarray()[0])[::-1][0:8]
  names = [[key for key, value in tfv.vocabulary_.items() if value == t_i][0] for t_i in top]
  print('   '+' '.join(names))

##LDA
## Compute our distribution of topics, as well as the distribution
## of words for each of those topics.
##

def lda_v(lda, text, features):
  return np.array([word_tokenize(text).count(f) for f in features])

def display_topics(model, feature_names, no_top_words):
    print("\nDisplaying LDA Topics")
    for topic_idx, topic in enumerate(model.components_):
        chosen = topic.argsort()[:-no_top_words - 1:-1]
        print("Topic %d:" % (topic_idx), \
              " ".join([feature_names[i] \
                        for i in chosen]))

# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf = DocX
tf_feature_names = vectorizer.get_feature_names()

#---------------
print("Calculating Dirichlet distribution of docs (LDA)")
# Run LDA
lda = LatentDirichletAllocation(n_components=no_topics, max_iter=5, \
                                learning_method='online', \
                                learning_offset=50., random_state=42)
tf_lda = lda.fit_transform(tf)

#no_top_words = 10

#display_topics(lda, tf_feature_names, 8)
#---------------
print("Documenting topics as weighted word distributions")
topics = []
for topic_idx, topic in enumerate(lda.components_):
  chosen = topic.argsort()[:-7:-1]
  topics.append(" ".join([tf_feature_names[i] \
                        for i in chosen]))
for idx, t in enumerate(topics):
  print(idx,t)

def lda2vec(lda, text, feature_names):
  vv = lda_v(lda, text, tf_feature_names)
  vv = vv.reshape(1,-1)
  return lda.transform(vv)

## Create word frequency

#---------------
print("Calculating interesting LDA word freqency in each sentence")
txt = ["This is my sample text"]
vv = CountVectorizer(vocabulary=tf_feature_names)
SentX = vv.fit_transform(SentText)
SentVec = lda.transform(SentX)

#---------------
print("Analyzing sentences for intersting-ness")
nDocs = np.max(doc_df['doc_id'].values)
nSents = SentX.shape[0]
SentStats = np.zeros((nSents,5))

for d_i in tqdm(range(nDocs)):
  df_i = sent_df[sent_df['doc_id']  == d_i]
  for idx, s_i in enumerate(df_i['sid'].values):
    toks = word_tokenize(df_i[df_i['sid'] == s_i]['text'].values[0])
    SentStats[s_i,0] = idx # sentence id sid
    SentStats[s_i,1] = len(toks) #all words
    SentStats[s_i,2] = np.sum(SentX[s_i,:])  #meaningful words
    SentStats[s_i,3] = SentStats[s_i,2]/(SentStats[s_i,1]+1) #relevance

info_m = np.mean(SentStats[:,3])
info_std = np.std(SentStats[:,3])
hi_locs = SentStats[np.argwhere(SentStats[:,3] > info_m+2*info_std),0].flatten()
lo_locs = SentStats[np.argwhere(SentStats[:,3] < info_m-1*info_std),0].flatten()
ok_locs = SentStats[np.argwhere(SentStats[:,3] > info_m+1*info_std),0].flatten()

#---------------
print("Choosing the most interesting sentences in each doc")
# Let's indicate which sentences we want to keep - relevant sentences
SentStats[:,4] = 0
SentStats[np.argwhere(SentStats[:,3] >  info_m+0*info_std),4] = 1

# OK, let's compute doc centroids
#---------------
print("Calculating the centroid for a doc's interesting sentences")
tensor_size = sent_df['tensor'].values[0].shape[0]
doc_centroids = np.zeros((tf_lda.shape[0], tensor_size))

def centroid(arr):
    length, dim = arr.shape
    return np.array([np.sum(arr[:, i])/length for i in range(dim)])

dead_docs  = []
for d_i in tqdm(range(nDocs)):
  vecs = []
  df_i = sent_df[sent_df.doc_id ==  d_i]
  for s_i in df_i['sid'].values:
    # only extract relevant sentences, longer than 3 words
    if (SentStats[s_i,4] > 0) and (SentStats[s_i,1] > 3):
      vecs.append(df_i['tensor'].values[0])
  if (len(vecs) < 1):
    # use null for dead docs
    dead_docs.append(d_i)
    vecs.append(np.zeros(tensor_size))
  vecs = np.array(vecs).reshape((len(vecs),tensor_size))
  doc_centroids[d_i] = centroid(vecs)

#concatenate both vectors, first the gamma scaled LDA
#encoding, then the BERT centroid for topical sentences

#---------------
print("Combining LDA and Bert centroid for each document")
gamma = 20
doc_both = np.c_[(gamma*tf_lda, doc_centroids)]

#---------------
print("Distilling the combined vector to 32 key dimensions")
# create an autoencoder to distill the document vector information
# to 3 dimensions
ae = Autoencoder(32)
ae.fit(doc_both)
# now predict new dense embeddings for every document
doc_dense = ae.encoder.predict(doc_both)

#---------------
print("Clustering the dense, 32-dim information vectors for every doc")
# cluster our documents using k-means
from sklearn.cluster import KMeans
doc_kmeans = KMeans(n_clusters=C, random_state=42).fit_transform(doc_dense)
doc_clusters = np.argmin(doc_kmeans,axis=1)
doc_centers = np.argmin(doc_kmeans,axis=0)

# well, how well did we do?

#---------------
print("Evaluating our technique")

ss_score = metrics.silhouette_score(doc_dense, doc_clusters, metric='euclidean')
ch_score = metrics.calinski_harabasz_score(doc_dense, doc_clusters)
cb_score = metrics.davies_bouldin_score(doc_dense, doc_clusters)

# [-1,1] higher is better, measure of separation using euc distance
print("Silhouette score:", ss_score)
#higher is better, tighter variance 
print("Calinski_harabasz:", ch_score)
#lower is better, for bigger, more separate clusters
print("Davies-Bouldin:",cb_score)

#---------------
print("Creating a breadth-first walk of all clusters, starting at centers")
# create a breadth-first-search feed for topics
doc_bfs = np.zeros(doc_kmeans.shape)
doc_bfs.fill(-1)
topic_counts = []

for c_i in range(doc_bfs.shape[1]):
  ci_docs = np.argwhere(doc_clusters == c_i)
  print("Cluster",c_i,"has",len(ci_docs),'docs')
  topic_counts.append(len(ci_docs))
  ci_doc_distances = doc_kmeans[ci_docs,c_i].flatten()
  for idx,ci_doc in enumerate(np.argsort(ci_doc_distances)): 
    doc_bfs[idx,c_i] = ci_docs[ci_doc]

feed = doc_bfs.flatten()
feed = np.array([np.int(f) for f in feed[feed >= 0]])

## Summarize cluster terms by calculating the mean
## of all docs in a cluster.
#---------------
print("Summarizing a cluster as the average of all TF/IDF values")
cluster_terms = []
for c_i in range(doc_bfs.shape[1]):
  ci_docs = np.argwhere(doc_clusters == c_i).flatten()
  avg_tfidf = np.array(np.mean(doc_tf[ci_docs],axis=0))
  best_terms = np.array(np.argsort(avg_tfidf[0,:]))[::-1][0:4] 
  names = [[key for key, value in tfv.vocabulary_.items() if value == t_i][0] for t_i in best_terms]
  cluster_terms.append(' '.join(names))

print(cluster_terms)

## Summarize cluster terms by calculating the mean
## of k docs closest to the centroid of each cluster.

#---------------
print("Summarizing a cluster as the average of 10 best documents and their TF/IDF values")
kcluster_terms = []
for c_i in range(doc_bfs.shape[1]):
  ci_docs = np.argwhere(doc_clusters == c_i).flatten()
  ci_doc_distances = doc_kmeans[ci_docs,c_i].flatten()
  closest_k_docs = ci_docs[np.argsort(ci_doc_distances)[0:10]]
  avg_tfidf = np.array(np.mean(doc_tf[closest_k_docs],axis=0))
  best_terms = np.array(np.argsort(avg_tfidf[0,:]))[::-1][0:4] 
  names = [[key for key, value in tfv.vocabulary_.items() if value == t_i][0] for t_i in best_terms]
  kcluster_terms.append(' '.join(names))

print(kcluster_terms)

#---------------
print("Uploading our feed for "+city_name+" as a breadth-first walk of clusters")
# Create a nice summary dataframe
tp = {'date': [],
      'index': [],
      'doc_id': [],
      'city': [],
      'z': [],
      'topic': [],
      'who':  [],
      'url': [],
      'title': [],
      'image': [],
      'distance': [],
      'snippet': []}

def create_feed(db, feed_ids, cluster_info):
  # we expect a list of doc_ids, one for
  # each topic, from topic 0 to topic C
  for idx in tqdm(range(len(feed_ids))):
    d_i = int(feed_ids[idx])
    #print("Hi",d_i,"there")
    df_i  = doc_df[doc_df.doc_id ==  d_i]
    df_j = sent_df[sent_df.doc_id == d_i]
    topic = cluster_info[d_i]
    sent_nums = df_j['sid'].values
    sent_relevance = SentStats[sent_nums,3]
    best_sentence = np.argmax(sent_relevance)
    snippet = ''
    if best_sentence > 0:
      snippet = df_j[df_j.sid == sent_nums[np.argmax(sent_relevance)]]['text'].values[0]
    db['date'].append(df_i['date'].values[0])
    db['index'].append(idx)
    db['doc_id'].append(d_i)
    db['topic'].append(topic)
    db['who'].append(df_i['domain'].values[0])
    db['z'].append(df_i['z'].values[0])
    db['url'].append(df_i['url'].values[0])
    db['city'].append(df_i['city'].values[0])
    db['title'].append(df_i['title'].values[0])
    db['image'].append(df_i['image_url'].values[0])
    db['distance'].append(doc_kmeans[d_i,topic])  # distance from this topic
    db['snippet'].append(snippet)

create_feed(tp,feed,doc_clusters)

#save our feed
from google.cloud import bigquery

client = bigquery.Client(project=project_id)
client.create_dataset('gdelt_sa',exists_ok=True)
table_id = project_id+'.gdelt_sa.daily_feed'

# clean slate
client.delete_table(table_id, not_found_ok=True) 

# Since string columns use the "object" dtype, pass in a (partial) schema
# to ensure the correct BigQuery data type.

job_config = bigquery.LoadJobConfig(schema=[
    bigquery.SchemaField("index", "INT64"),
    bigquery.SchemaField("date", "INT64"),
    bigquery.SchemaField("doc_id", "INT64"),
    bigquery.SchemaField("city","STRING"),
    bigquery.SchemaField("z", "FLOAT64"),
    bigquery.SchemaField("topic", "INT64"),
    bigquery.SchemaField("who", "STRING"),
    bigquery.SchemaField("url", "STRING"),
    bigquery.SchemaField("title", "STRING"),
    bigquery.SchemaField("image", "STRING"),
    bigquery.SchemaField("distance", "FLOAT64"),
    bigquery.SchemaField("snippet", "STRING")
])

feed_df = pd.DataFrame(data=tp)

job = client.load_table_from_dataframe(
    feed_df, table_id, job_config=job_config
)

# Wait for the load job to complete.
job.result()

#---------------
print("Uploading the cluster descriptions to BigQuery for "+city_name)
#save our topics as 'themes' that are used to build high-order topics for users
table_id = project_id+'.gdelt_sa.themes'

# clean slate
client.delete_table(table_id, not_found_ok=True) 

# Since string columns use the "object" dtype, pass in a (partial) schema
# to ensure the correct BigQuery data type.

top_dict = {'index': [], 'name': [], 'city': []}
for idx,name in enumerate(kcluster_terms):
  top_dict['index'].append(idx)
  top_dict['name'].append(name)
  top_dict['city'].append(city_name)

topic_df = pd.DataFrame(data=top_dict)

job_config = bigquery.LoadJobConfig(schema=[
    bigquery.SchemaField("index", "INT64"),
    bigquery.SchemaField("name", "STRING"),
    bigquery.SchemaField("city", "STRING")
])

job = client.load_table_from_dataframe(
    topic_df, table_id, job_config=job_config
)

# Wait for the load job to complete.
job.result()

#---------------
print("Finis!")