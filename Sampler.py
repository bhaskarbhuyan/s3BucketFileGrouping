from fuzzywuzzy import fuzz
from fuzzywuzzy import process
# import boto3
# from boto3.session import Session

import numpy as np
import sklearn.cluster
import distance

#Get S3 Files in a list

s3filelist = list()

ACCESS_KEY='xxxx'
SECRET_KEY='xxxx'

session = Session(aws_access_key_id=ACCESS_KEY,
                  aws_secret_access_key=SECRET_KEY)
s3 = session.resource('s3')
my_bucket = s3.Bucket('Bucket_name') #replace bucket_name  with actual bucket
for file in my_bucket.objects.filter(Prefix="folder_name/"):
    s3filelist.append(file.rstrip().lstrip())

#Now s3 filelist have all the filenames




#Group all the files in s3 into similar groups. There will e a key word and a list of words similar to the key word
matchDict = dict()
otherWordsinCluster = list()

# words = ''' Any String '''.split(" ") #Replace this line

# wordList = np.asarray(words)

wordList = s3filelist

wordList = np.asarray(wordList) #So that indexing with a list will work

lev_similarity = -1*np.array([[distance.levenshtein(w1,w2) for w1 in wordList] for w2 in wordList])

affprop = sklearn.cluster.AffinityPropagation(affinity="precomputed", damping=0.5)
affprop.fit(lev_similarity)
for cluster_id in np.unique(affprop.labels_):
    masterWord = wordList[affprop.cluster_centers_indices_[cluster_id]]
    cluster = np.unique(wordList[np.nonzero(affprop.labels_==cluster_id)])
    otherWordsinCluster = ", ".join(cluster)
    # print(" - *%s:* %s" % (masterWord, otherWordsinCluster))
    matchDict.update({masterWord:otherWordsinCluster})




#iterate over the cluster of words to and add fuzzy logic to match the words again and find out the matches less than 60, these will be the odd matches


oddWords = []
for queryword, otherwords in matchDict.items():
    # print(queryword,otherwords)
    otherwords = otherwords.rstrip().lstrip().split(" ")
    # print(otherwords)
    x = process.extract(queryword, otherwords)
    # print(x)
    for item in x:
        if item[1] < 60:
            oddWords.append(item[0])
    # print(oddWords)
    # oddWords = []

print(oddWords)