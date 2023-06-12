import pyspark, sys, time, csv, math,json
import xgboost as xgb
import pandas as pd
import datetime
def avg(x):
    list(x)
    return sum(x) / len(x)

if __name__ == '__main__':
    start=time.time()
    folder_path='./data'
    test_file='./yelp_val.csv'
    output_file='./output2_2.csv'

    """folder_path =sys.argv[1]
    test_file =sys.argv[2]
    output_file =sys.argv[3]"""

    sc = pyspark.SparkContext('local[*]', 'task2_2')
    sc.setLogLevel("WARN")

    user_rdd = sc.textFile(folder_path+'/user.json').map(lambda row: json.loads(row))
    business_rdd = sc.textFile(folder_path + '/business.json').map(lambda row: json.loads(row))
    yelp_train = sc.textFile(folder_path+'/yelp_train.csv').filter(lambda row: not row.startswith("user_id")) \
        .map(lambda row: row.split(','))
    yelp_test = sc.textFile(test_file).filter(lambda row: not row.startswith("user_id")) \
        .map(lambda row: row.split(','))

    avg_user_review_count=user_rdd.map(lambda row:row['review_count']).mean()
    avg_user_average_stars = user_rdd.map(lambda row: row['average_stars']).mean()
    avg_user_fans=user_rdd.map(lambda row:row['fans']).mean()
    avg_user_useful=user_rdd.map(lambda row:row['useful']).mean()
    avg_business_stars=business_rdd.map(lambda row:row['stars']).mean()
    avg_business_review_count=business_rdd.map(lambda row:row['review_count']).mean()

    #print(business_rdd.take(10))
    user_feature=user_rdd.map(lambda row:(row['user_id'],(row['review_count'],row['average_stars'],row['fans'],row['useful'])))
    business_feature=business_rdd.map(lambda row:(row['business_id'],(row['stars'],row['review_count'])))
    train=yelp_train.map(lambda row:(row[0],(row[1],row[2])))
    #userid,business_id,user_review_count,user_average_stars,stars
    def removeNA1(row):
        if row[1][1][0]==None:
            row[1][1][0]=avg_user_review_count
        if row[1][1][1]==None:
            row[1][1][1]=avg_user_average_stars
        if row[1][1][2]==None:
            row[1][1][2]=avg_user_fans
        if row[1][1][3]==None:
            row[1][1][3]=avg_user_useful
        return row
    train=train.leftOuterJoin(user_feature).map(lambda row:removeNA1(row)).map(lambda row:(row[0],row[1][0][0],row[1][1][0],row[1][1][1],row[1][1][2],row[1][1][3],float(row[1][0][1])))
    #business_id,(user_id,user_review_count,user_average_stars,fans,useful,stars)
    train=train.map(lambda row :(row[1],(row[0],row[2],row[3],row[4],row[5],row[6])))
    #user_id,business_id,user_review_count,user_average_stars,fans,business_average_star,business_review_count, actual_star
    def removeNA2(row):
        if row[1][1][0]==None:
            row[1][1][0]=avg_business_stars
        if row[1][1][1]==None:
            row[1][1][1] = avg_business_review_count
        return row

    train=train.leftOuterJoin(business_feature).map(lambda row:removeNA2(row)).map(lambda row:(row[1][0][0],row[0],row[1][0][1],row[1][0][2],row[1][0][3],row[1][0][4],row[1][1][0],row[1][1][1],row[1][0][5]))

    test = yelp_test.map(lambda row: (row[0], row[1]))
    # userid,business_id,user_review_count,user_average_stars
    #print(test.leftOuterJoin(user_feature))
    test = test.leftOuterJoin(user_feature).map(lambda row:removeNA1(row)).map(lambda row: (row[0], row[1][0], row[1][1][0], row[1][1][1],row[1][1][2],row[1][1][3]))
    # business_id,(user_id,user_review_count,user_average_stars)
    test = test.map(lambda row: (row[1], (row[0], row[2], row[3], row[4],row[5])))
    # user_id,business_id,user_review_count,user_average_stars,business_average_star,business_review_count, actual_star
    #print(test.leftOuterJoin(business_feature).collect())
    test = test.leftOuterJoin(business_feature).map(lambda row:removeNA2(row)).map(lambda row: (row[1][0][0], row[0], row[1][0][1], row[1][0][2],row[1][0][3],row[1][0][4],row[1][1][0], row[1][1][1]))

    train_x=train.map(lambda row:[row[2],row[3],row[4],row[5],row[6],row[7]]).collect()
    train_y=train.map(lambda row:row[8]).collect()
    test_x=test.map(lambda row:[row[2],row[3],row[4],row[5],row[6],row[7]]).collect()
    train_x_df = pd.DataFrame(train_x, columns=['user_review_count','user_average_stars','fans','useful','business_average_star','business_review_count'])
    test_x_df = pd.DataFrame(test_x, columns=['user_review_count', 'user_average_stars', 'fans','useful','business_average_star','business_review_count'])
    xgb_model = xgb.XGBRegressor(n_estimators=500,max_depth=5)
    xgb_model.fit(train_x,train_y)
    y_pred = xgb_model.predict(test_x)
    test_pair=test.map(lambda row:(row[0],row[1])).collect()
    result_dict={}
    for i in range(len(test_pair)):
        result_dict[test_pair[i]]=y_pred[i]

    result=yelp_test.map(lambda row:[row[0],row[1],result_dict[(row[0],row[1])]])


    """RMSE=0
    for i in range(1, len(test_y)):
        print(y_pred[i],y_pred[i]*1)
        RMSE += ((float(y_pred[i]) - float(test_y[i])) * (float(y_pred[i]) - float(test_y[i])))
    RMSE = math.sqrt(RMSE / (len(test_y)))
    print(RMSE)"""

    title_csv = ["user_id", "business_id", "prediction"]
    with open(output_file, 'w') as f:
        # using csv.writer method from CSV package
        write = csv.writer(f)
        write.writerow(title_csv)
        write.writerows(result.collect())

    end=time.time()
    print("Duration:{0:.2f} ".format(end - start))