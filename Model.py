# MUSIC RECOMMENDER SYSTEM
import numpy as np
import pandas
from sklearn.model_selection import train_test_split
import  time
from sklearn.externals import joblib
import pylab as pl
import Recommenders as Recommenders
import Evaluation as Evaluation
import pylab as pl


triplets_file = '10000.txt'
songs_metadata_file = 'song_data.csv'

song_df_1 = pandas.read_table(triplets_file, header=None)
song_df_1.columns = ['user_id', 'song_id', 'listen_count']
song_df_2 = pandas.read_csv(songs_metadata_file)
song_df = pandas.merge(song_df_1, song_df_2.drop_duplicates(['song_id']), on="song_id", how="left")
#print(song_df.head())
song_df = song_df.head(10000)
song_df['song'] = song_df['title'].map(str) + " - " + song_df['artist_name']

song_grouped = song_df.groupby(['song']).agg({'listen_count': 'count'}).reset_index()
grouped_sum = song_grouped['listen_count'].sum()
song_grouped['percentage']  = song_grouped['listen_count'].div(grouped_sum)*100
song_grouped.sort_values(['listen_count', 'song'], ascending = [0,1])
#print(song_grouped.head())

users = song_df['user_id'].unique()
#print(len(users))
songs = song_df['song'].unique()
#print(len(songs))

#CREATE A RECOMMENDER
train_data, test_data = train_test_split(song_df, test_size = 0.20, random_state=0)
#print(train_data.head(5))


#USING RECOMMENDERS.PY AS A BLACKBOX

#for a particular user_id
pm = Recommenders.popularity_recommender_py()
pm.create(train_data, 'user_id', 'song')
user_id = users[5]
#print(pm.recommend(user_id))

#PERSONALIZED RECOMMENDER SYSTEM
is_model = Recommenders.item_similarity_recommender_py()
is_model.create(train_data, 'user_id', 'song')

user_id = users[5]
user_items = is_model.get_user_items(user_id)

print("------------------------------------------------------------------------------------")
print("Training data songs for the user userid: %s:" % user_id)
print("------------------------------------------------------------------------------------")

for user_item in user_items:
    print(user_item)

print("----------------------------------------------------------------------")
print("Recommendation process going on:")
print("----------------------------------------------------------------------")
###Recommend songs for the user using personalized model
#print(is_model.recommend(user_id))


#  REMOVE COMMENTS TO THE CODE ABOVE TO GET UR RESULT !!
#SIMILAR SONGS IN DATASET
#print(is_model.get_similar_items(['Yellow - Coldplay']))



## QUANTITVE COMPARISON BETWEEN MODELS
#EVALUATION>PY USED AS A BLACKBOX

start = time.time()
user_sample = 0.05
pr = Evaluation.precision_recall_calculator(test_data, train_data, pm, is_model)
(pm_avg_precision_list, pm_avg_recall_list, ism_avg_precision_list, ism_avg_recall_list) = pr.calculate_measures(user_sample)
end = time.time()
print(end - start)

def plot_precision_recall(m1_precision_list, m1_recall_list, m1_label, m2_precision_list, m2_recall_list, m2_label):
    pl.clf()
    pl.plot(m1_recall_list, m1_precision_list, label=m1_label)
    pl.plot(m2_recall_list, m2_precision_list, label=m2_label)
    pl.xlabel('Recall')
    pl.ylabel('Precision')
    pl.ylim([0.0, 0.20])
    pl.xlim([0.0, 0.20])
    pl.title('Precision-Recall curve')
    #pl.legend(loc="upper right")
    pl.legend(loc=9, bbox_to_anchor=(0.5, -0.2))
    pl.show()

print("Plotting precision recall curves.")

plot_precision_recall(pm_avg_precision_list, pm_avg_recall_list, "popularity_model",
                      ism_avg_precision_list, ism_avg_recall_list, "item_similarity_model")



print("Plotting precision recall curves for a larger subset of data (100,000 rows) (user sample = 0.005).")
#Read the persisted files
pm_avg_precision_list = joblib.load('pm_avg_precision_list_3.pkl')
pm_avg_recall_list = joblib.load('pm_avg_recall_list_3.pkl')
ism_avg_precision_list = joblib.load('ism_avg_precision_list_3.pkl')
ism_avg_recall_list = joblib.load('ism_avg_recall_list_3.pkl')
print("Plotting precision recall curves.")
plot_precision_recall(pm_avg_precision_list, pm_avg_recall_list, "popularity_model",
                      ism_avg_precision_list, ism_avg_recall_list, "item_similarity_model")


print("Plotting precision recall curves for a larger subset of data (100,000 rows) (user sample = 0.005).")
pm_avg_precision_list = joblib.load('pm_avg_precision_list_2.pkl')
pm_avg_recall_list = joblib.load('pm_avg_recall_list_2.pkl')
ism_avg_precision_list = joblib.load('ism_avg_precision_list_2.pkl')
ism_avg_recall_list = joblib.load('ism_avg_recall_list_2.pkl')
print("Plotting precision recall curves.")
plot_precision_recall(pm_avg_precision_list, pm_avg_recall_list, "popularity_model",
                      ism_avg_precision_list, ism_avg_recall_list, "item_similarity_model")

