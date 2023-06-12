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
    output_file='./output2_3.csv'

    """folder_path =sys.argv[1]
    test_file =sys.argv[2]
    output_file =sys.argv[3]"""

    sc = pyspark.SparkContext('local[*]', 'task2_3')
    sc.setLogLevel("WARN")

    user_rdd = sc.textFile(folder_path+'/user.json').map(lambda row: json.loads(row))
    business_rdd = sc.textFile(folder_path + '/business.json').map(lambda row: json.loads(row))
    yelp_train = sc.textFile(folder_path+'/yelp_train.csv').filter(lambda row: not row.startswith("user_id")) \
        .map(lambda row: row.split(',')).map(lambda row: [row[0], row[1], float(row[2])])
    yelp_test = sc.textFile(test_file).filter(lambda row: not row.startswith("user_id")) \
        .map(lambda row: row.split(','))
    #Model Based
    avg_user_review_count = user_rdd.map(lambda row: row['review_count']).mean()
    avg_user_average_stars = user_rdd.map(lambda row: row['average_stars']).mean()
    avg_user_fans = user_rdd.map(lambda row: row['fans']).mean()
    avg_user_useful = user_rdd.map(lambda row: row['useful']).mean()
    avg_business_stars = business_rdd.map(lambda row: row['stars']).mean()
    avg_business_review_count = business_rdd.map(lambda row: row['review_count']).mean()

    # print(business_rdd.take(10))
    user_feature = user_rdd.map(
        lambda row: (row['user_id'], (row['review_count'], row['average_stars'], row['fans'], row['useful'])))
    business_feature = business_rdd.map(lambda row: (row['business_id'], (row['stars'], row['review_count'])))
    train = yelp_train.map(lambda row: (row[0], (row[1], row[2])))


    # userid,business_id,user_review_count,user_average_stars,stars
    def removeNA1(row):
        if row[1][1][0] == None:
            row[1][1][0] = avg_user_review_count
        if row[1][1][1] == None:
            row[1][1][1] = avg_user_average_stars
        if row[1][1][2] == None:
            row[1][1][2] = avg_user_fans
        if row[1][1][3] == None:
            row[1][1][3] = avg_user_useful
        return row


    train = train.leftOuterJoin(user_feature).map(lambda row: removeNA1(row)).map(
        lambda row: (row[0], row[1][0][0], row[1][1][0], row[1][1][1], row[1][1][2], row[1][1][3], float(row[1][0][1])))
    # business_id,(user_id,user_review_count,user_average_stars,fans,useful,stars)
    train = train.map(lambda row: (row[1], (row[0], row[2], row[3], row[4], row[5], row[6])))


    # user_id,business_id,user_review_count,user_average_stars,fans,business_average_star,business_review_count, actual_star
    def removeNA2(row):
        if row[1][1][0] == None:
            row[1][1][0] = avg_business_stars
        if row[1][1][1] == None:
            row[1][1][1] = avg_business_review_count
        return row


    train = train.leftOuterJoin(business_feature).map(lambda row: removeNA2(row)).map(lambda row: (
    row[1][0][0], row[0], row[1][0][1], row[1][0][2], row[1][0][3], row[1][0][4], row[1][1][0], row[1][1][1],
    row[1][0][5]))

    test = yelp_test.map(lambda row: (row[0], row[1]))
    # userid,business_id,user_review_count,user_average_stars
    # print(test.leftOuterJoin(user_feature))
    test = test.leftOuterJoin(user_feature).map(lambda row: removeNA1(row)).map(
        lambda row: (row[0], row[1][0], row[1][1][0], row[1][1][1], row[1][1][2], row[1][1][3]))
    # business_id,(user_id,user_review_count,user_average_stars)
    test = test.map(lambda row: (row[1], (row[0], row[2], row[3], row[4], row[5])))
    # user_id,business_id,user_review_count,user_average_stars,business_average_star,business_review_count, actual_star
    # print(test.leftOuterJoin(business_feature).collect())
    test = test.leftOuterJoin(business_feature).map(lambda row: removeNA2(row)).map(lambda row: (
    row[1][0][0], row[0], row[1][0][1], row[1][0][2], row[1][0][3], row[1][0][4], row[1][1][0], row[1][1][1]))

    train_x = train.map(lambda row: [row[2], row[3], row[4], row[5], row[6], row[7]]).collect()
    train_y = train.map(lambda row: row[8]).collect()
    test_x = test.map(lambda row: [row[2], row[3], row[4], row[5], row[6], row[7]]).collect()
    train_x_df = pd.DataFrame(train_x, columns=['user_review_count', 'user_average_stars', 'fans', 'useful',
                                                'business_average_star', 'business_review_count'])
    test_x_df = pd.DataFrame(test_x, columns=['user_review_count', 'user_average_stars', 'fans', 'useful',
                                              'business_average_star', 'business_review_count'])
    xgb_model = xgb.XGBRegressor()
    xgb_model.fit(train_x,train_y)
    y_pred = xgb_model.predict(test_x)
    # dictionary pair:model based prediction
    test_pair=test.map(lambda row:(row[0],row[1])).collect()
    model_pred={}
    for i in range(len(test_pair)):
        model_pred[test_pair[i]]=y_pred[i]

    #Item Based CF
    neighbour_dict={}
    yelp_u_b = yelp_train.map(lambda row: (row[0], row[1])).groupByKey().mapValues(list).collectAsMap()
    yelp_b_u = yelp_train.map(lambda row: (row[1], row[0])).groupByKey().mapValues(set).collectAsMap()
    yelp_pair_rate = yelp_train.map(lambda row: [(row[0], row[1]), row[2]]).collectAsMap()
    avg_all = yelp_train.map(lambda row: row[2]).mean()
    avg_rating_item = yelp_train.map(lambda row: (row[1], float(row[2]))).groupByKey().mapValues(avg).collectAsMap()
    avg_rating_user = yelp_train.map(lambda row: (row[0], float(row[2]))).groupByKey().mapValues(avg).collectAsMap()
    weight_dict = {}
    def weight_calculation(b1, b2):
        u1 = yelp_b_u[b1]
        u2 = yelp_b_u[b2]
        n = 0
        d1 = 0
        d2 = 0
        u = list(u1.intersection(u2))
        # if no common user
        if len(u) == 0:
            mean_diff = abs(avg_rating_item[b1] - avg_rating_item[b2])
            if mean_diff <= 0.85:
                return 1
            elif mean_diff <= 2:
                return 0.85
            elif mean_diff <= 3:
                return 0.5
            elif mean_diff <= 4:
                return 0.25
            else:
                return 0
        for i in range(len(u)):
            pair1 = (u[i], b1)
            pair2 = (u[i], b2)
            avg_b1 = avg_rating_item[b1]
            avg_b2 = avg_rating_item[b2]
            rate_uib1 = yelp_pair_rate[pair1]
            rate_uib2 = yelp_pair_rate[pair2]
            n = n + ((rate_uib1 - avg_b1) * (rate_uib2 - avg_b2))
            d1 = d1 + ((rate_uib1 - avg_b1) * (rate_uib1 - avg_b1))
            d2 = d2 + ((rate_uib2 - avg_b2) * (rate_uib2 - avg_b2))

        if (math.sqrt(d1) * math.sqrt(d2)) == 0:
            return 0
        # print(len(u))
        """if len(u) <= 20:
            return 0.2*(n / (math.sqrt(d1) * math.sqrt(d2)))"""
        return n / (math.sqrt(d1) * math.sqrt(d2))
    def IB_predict(user, business):

        p = 0
        # for all business this user is rated
        if (user not in yelp_u_b) and (business not in yelp_b_u):
            return avg_all
        if user not in yelp_u_b:
            return avg_rating_item[business]
        if business not in yelp_b_u:
            return avg_rating_user[user]

        b_l = yelp_u_b[user]
        rate_weight_t = 0
        weight_t = 0
        weight_list = []
        for i in range(len(b_l)):
            pair = (b_l[i], business)
            if pair not in weight_dict:
                w = weight_calculation(b_l[i], business)
                # w = w * pow(abs(w), (2.5 - 1))
                weight_dict[pair] = w
            else:
                w = weight_dict[pair]
            if w > 0:
                weight_list.append((b_l[i], w))
        if len(weight_list) == 0:
            return (avg_rating_item[business] + avg_rating_user[user]) / 2
        weight_list.sort(key=lambda item: item[1], reverse=True)
        neighbour_dict[(user,business)]=len(weight_list)
        #print(neighbour_dict)
        for i in range(len(weight_list)):
            b_c = weight_list[i][0]
            w = weight_list[i][1]
            # print(user, business)
            rate_weight_t = rate_weight_t + (w * yelp_pair_rate[(user, b_c)])
            weight_t = weight_t + abs(w)
            # print(b_l[i],w)
        if weight_t == 0:
            p = 0
        else:
            p = rate_weight_t / weight_t
        # p = avg_rating_user[user]
        return p
    yelp_predict = yelp_test.map(lambda row: (row[0], row[1], IB_predict(row[0], row[1])))
    IB_pred=yelp_predict.map(lambda row:((row[0],row[1]),row[2])).collectAsMap()
    def hybrid_res(row):
        pair=(row[0],row[1])
        #user_review_count, user_average_stars, business_average_star, business_review_count=row[2],row[3],row[4],row[5]
        alpha=1.0
        #print(model_pred[pair],1*model_pred[pair],1.0*model_pred[pair])
        predict_combine=(alpha*model_pred[pair])+((1.0-alpha)*IB_pred[pair])
        #print(predict_combine)
        return predict_combine
    result=yelp_test.map(lambda row:(row[0],row[1],hybrid_res(row)))
    y_pred_f=result.map(lambda row:row[2]).collect()

    title_csv = ["user_id", "business_id", "prediction"]

    with open(output_file, 'w') as f:
        # using csv.writer method from CSV package
        write = csv.writer(f)
        write.writerow(title_csv)
        write.writerows(result.collect())

    end=time.time()
    print("Duration:{0:.2f} ".format(end - start))