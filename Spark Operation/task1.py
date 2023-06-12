# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import pyspark,sys, json, operator, re

def total_number_review(rdd):
    return rdd.map(lambda row: row["review_id"]).distinct().count()

def printf(part):
    print(list(part))

def review_number_2018(rdd):
    return rdd.filter(lambda row: '2018' in row["date"]).count()
def number_user(rdd):
    return rdd.map(lambda row: row["user_id"]).distinct().count()
def top10(rdd):
    #print(rdd.map(lambda row: (row["user_id"],1)).reduceByKey(lambda a,b:a+b).takeOrdered(10, key=lambda row: [-row[1],row[0]]))
    return rdd.map(lambda row: (row["user_id"],1)).reduceByKey(lambda a,b:a+b).takeOrdered(10, key=lambda row: [-row[1],row[0]])
def number_business(rdd):
    return rdd.map(lambda row: row["business_id"]).distinct().count()
def top10b(rdd):
    return rdd.map(lambda row: (row["business_id"],1)).reduceByKey(lambda a,b:a+b).takeOrdered(10, key=lambda row: [-row[1],row[0]])
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #input_path=sys.argv[1]
    #output_path=sys.argv[2]
    input_path='./review.json'
    output_path='./out-task1.json'
    result={}

    sc=pyspark.SparkContext('local[*]','task1')
    review_rdd=sc.textFile(input_path).map(lambda row: json.loads(row))
    print(review_rdd.collect())
    #a.The total number of reviews
    result['n_review']=total_number_review(review_rdd)
    print('The total number of reviews : ', result['n_review'])
    #b.The number of reviews in 2018
    result['n_review_2018']=review_number_2018(review_rdd)
    print('The number of reviews in 2018: ',result['n_review_2018'])
    #c.The number of distinct users who wrote reviews
    result['n_user']=number_user(review_rdd)
    print('The number of distinct users who wrote reviews:', result['n_user'])
    #The top 10 users who wrote the largest numbers of reviews and the number of reviews they wrote
    result['top10_user']=top10(review_rdd)
    print('The top 10 users who wrote the largest numbers of reviews and the number of reviews they wrote:',result['top10_user'])
    #The number of distinct businesses that have been reviewed
    result['n_business']=number_business(review_rdd)
    print('The number of distinct businesses that have been reviewed',result['n_business'])
    #The top 10 businesses that had the largest numbers of reviews and the number of reviews they had
    result['top10_business']=top10b(review_rdd)
    print('The top 10 businesses that had the largest numbers of reviews and the number of reviews they had',result['top10_business'])

    with open(output_path, 'w') as o:
        json.dump(result, o, sort_keys=False)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
