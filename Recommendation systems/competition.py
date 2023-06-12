"""
Method Description:
In the competition, I add more variables to the dataset,such as the number of friends of each user,
whether the  business can make a reservation, whether a user tips the business or not,
and the number of photo that business has. Also, since most variables I added are categorical,
I use catboost regressor instead of xgbregressor.

Error Distribution:
>=0 and <1: 102458
>=1 and <2: 32643
>=2 and <3: 6093
>=3 and <4: 850
>=4: 1

RMSE:
0.975442074639811


Execution Time:
245.73s
"""
import pyspark, sys, time, csv, math,json
#import xgboost as xgb
import pandas as pd
import catboost as cb
#0.9802354320782038 2000 6

def avg(x):
    list(x)
    return sum(x) / len(x)

if __name__ == '__main__':
    start=time.time()
    folder_path='./data'
    test_file='./yelp_val_in.csv'
    output_file='./output2_5.csv'

    """folder_path =sys.argv[1]
    test_file =sys.argv[2]
    output_file =sys.argv[3]"""

    sc = pyspark.SparkContext('local[*]', 'task2_2')
    sc.setLogLevel("WARN")

    user_rdd = sc.textFile(folder_path+'/user.json').map(lambda row: json.loads(row))
    business_rdd = sc.textFile(folder_path + '/business.json').map(lambda row: json.loads(row))
    tips_rdd=sc.textFile(folder_path + '/tip.json').map(lambda row: json.loads(row))
    photo_rdd=sc.textFile(folder_path + '/photo.json').map(lambda row: json.loads(row))

    tips_feature=tips_rdd.map(lambda row:[(row['user_id'],row['business_id']),1]).groupByKey().mapValues(len).collectAsMap()
    tips_like = tips_rdd.map(lambda row: [(row['user_id'], row['business_id']), row['likes']]).groupByKey().mapValues(list).map(lambda row:[row[0],sum(row[1])]).collectAsMap()

    photo_feature=photo_rdd.map(lambda row:(row['business_id'],1)).groupByKey().mapValues(len).collectAsMap()
    #print(photo_feature)
    yelp_train = sc.textFile(folder_path+'/yelp_train.csv').filter(lambda row: not row.startswith("user_id")) \
        .map(lambda row: row.split(','))
    yelp_test = sc.textFile(test_file).filter(lambda row: not row.startswith("user_id")) \
        .map(lambda row: row.split(','))

    """
    train_df = pd.DataFrame(yelp_train.collect(), columns=['user_id', 'business_id', 'stars'])
    test_df = pd.DataFrame(yelp_test.collect(), columns=['user_id', 'business_id'])
    test_df['stars'] = 0
    reader: Reader = Reader()
    data = Dataset.load_from_df(train_df, reader)
    test_data = Dataset.load_from_df(test_df, reader)
    test_set = test_data.build_full_trainset().build_testset()
    train_set = data.build_full_trainset()

    svd_model = SVD(n_factors=100, n_epochs=50, lr_all=0.005, reg_all=0.1)
    svd_model.fit(train_set)
    result_svd_test= yelp_test.map(lambda row: [row[0], row[1], svd_model.predict(uid=row[0], iid=row[1]).est])
    SVD_test=result_svd_test.map(lambda row: row[2]).collect()
    result_svd_train = yelp_train.map(lambda row: [row[0], row[1], svd_model.predict(uid=row[0], iid=row[1]).est])
    SVD_train = result_svd_train.map(lambda row: row[2]).collect()
    """

    avg_user_review_count=user_rdd.map(lambda row:row['review_count']).mean()
    avg_user_average_stars = user_rdd.map(lambda row: row['average_stars']).mean()
    avg_user_fans=user_rdd.map(lambda row:row['fans']).mean()
    avg_user_useful=user_rdd.map(lambda row:row['useful']).mean()
    avg_business_stars=business_rdd.map(lambda row:row['stars']).mean()
    avg_business_review_count=business_rdd.map(lambda row:row['review_count']).mean()
    def friend_count(string):
        if string=='None':
            return 0
        li = len(list(string.split(",")))
        return li
    #print(business_rdd.take(10))
    user_feature=user_rdd.map(lambda row:(row['user_id'],(row['review_count'],row['average_stars'],row['fans'],row['useful'],row['funny'],row['cool'],friend_count(row['friends']),row['compliment_hot'],row['compliment_note'],row['compliment_photos'])))

    def check_reservation(dic):
        if dic is None:
            return 0
        if 'RestaurantsReservations' in dic:
            if dic['RestaurantsReservations']==True:
                return 2
            else:
                return 1
        else:
            return 0
    def check_creditcard(dic):
        if dic is None:
            return 0
        if 'BusinessAcceptsCreditCards' in dic:
            if dic['BusinessAcceptsCreditCards']==True:
                return 2
            else:
                return 1
        else:
            return 0
    def good_for_kid(dic):
        if dic is None:
            return 0
        if 'GoodForKids' in dic:
            if dic['GoodForKids']==True:
                return 2
            else:
                return 1
        else:
            return 0

    def bike_park(dic):
        if dic is None:
            return 0
        if 'BikeParking' in dic:
            if dic['BikeParking']==True:
                return 2
            else:
                return 1
        else:
            return 0

    def caters(dic):
        if dic is None:
            return 0
        if 'Caters' in dic:
            if dic['Caters']==True:
                return 2
            else:
                return 1
        else:
            return 0


    def noise(dic):
        if dic is None:
            return 0
        if 'NoiseLevel' in dic:
            if dic['NoiseLevel']=='quiet':
                return 3
            elif dic['NoiseLevel']=='average':
                return 2
            else:
                return 1
        else:
            return 0


    def price_range(dic):
        if dic is None:
            return 0
        if 'RestaurantsPriceRange2' in dic:
            return dic['RestaurantsPriceRange2']
        else:
            return 0

    def if_resturant(row):
        if 'categories' not in row:
            return 0
        else:
            if row['categories'] is None:
                return 0
            str1=row['categories']
            l= list(str1.split(","))
            if 'Restaurants' in l:
                return 2
            else:
                return 1
    def if_hotels(row):
        if 'categories' not in row:
            return 0
        else:
            if row['categories'] is None:
                return 0
            str1=row['categories']
            l= list(str1.split(","))
            if 'Hotels' in l:
                return 2
            else:
                return 1
    def if_shopping(row):
        if 'categories' not in row:
            return 0
        else:
            if row['categories'] is None:
                return 0
            str1=row['categories']
            l= list(str1.split(","))
            if 'Shopping' in l:
                return 2
            else:
                return 1



    business_feature=business_rdd.map(lambda row:(row['business_id'],(row['stars'],row['review_count'],row['is_open'],check_reservation(row['attributes']),check_creditcard(row['attributes']),row['longitude'],row['latitude'],price_range(row['attributes']),if_resturant(row),good_for_kid(row['attributes']),noise(row['attributes']))))
    train=yelp_train.map(lambda row:(row[0],(row[1],row[2])))
    #userid, business_id, user_review_count, user_average_stars, fans, useful, funny, cool, friend_count, compliment_hot, compliment_note, compliment_photos, stars
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
    train=train.leftOuterJoin(user_feature).map(lambda row:removeNA1(row)).map(lambda row:(row[0],row[1][0][0],row[1][1][0],row[1][1][1],row[1][1][2],row[1][1][3],row[1][1][4],row[1][1][5],row[1][1][6],row[1][1][7],row[1][1][8],row[1][1][9],float(row[1][0][1])))
    #business_id,(user_id,user_review_count,user_average_stars,fans,useful, funny, cool, friend_count, compliment_hot, compliment_note, compliment_photos,stars)
    train=train.map(lambda row :(row[1],(row[0],row[2],row[3],row[4],row[5],row[6],row[7],row[8],row[9],row[10],row[11],row[12])))
    #user_id,business_id,user_review_count,user_average_stars,fans,business_average_star,business_review_count, actual_star
    #userid, business_id, user_review_count, user_average_stars, fans, useful, funny, cool, friend_count, compliment_hot, compliment_note, compliment_photos, business_average_star, business_review_count ,business_open, reservation, credit_card, stars
    def removeNA2(row):
        if row[1][1][0]==None:
            row[1][1][0]=avg_business_stars
        if row[1][1][1]==None:
            row[1][1][1] = avg_business_review_count
        return row
    train=train.leftOuterJoin(business_feature).map(lambda row:removeNA2(row)).map(lambda row:(row[1][0][0],row[0],row[1][0][1],row[1][0][2],row[1][0][3],row[1][0][4],row[1][0][5],row[1][0][6],row[1][0][7],row[1][0][8],row[1][0][9],row[1][0][10],row[1][1][0],row[1][1][1],row[1][1][2],row[1][1][3],row[1][1][4],row[1][1][5],row[1][1][6],row[1][1][7],row[1][1][8],row[1][1][9],row[1][1][10],row[1][0][11]))
    def add_feature(row):
        l=list(row)
        user_id=row[0]
        business_id=row[1]
        u_b_pair=(user_id,business_id)
        if u_b_pair in tips_feature:
            l.append(tips_feature[u_b_pair])
        else:
            l.append(0)
        if business_id in photo_feature:
            l.append(photo_feature[business_id])
        else:
            l.append(0)
        return l

    train=train.map(lambda row:add_feature(row))

    #process test data
    test = yelp_test.map(lambda row: (row[0], row[1]))
    # userid,business_id,user_review_count,user_average_stars

    test = test.leftOuterJoin(user_feature).map(lambda row:removeNA1(row)).map(lambda row:(row[0],row[1][0],row[1][1][0],row[1][1][1],row[1][1][2],row[1][1][3],row[1][1][4],row[1][1][5],row[1][1][6],row[1][1][7],row[1][1][8],row[1][1][9]))
    # business_id,(user_id,user_review_count,user_average_stars)
    test = test.map(lambda row: (row[1], (row[0], row[2], row[3], row[4],row[5],row[6],row[7],row[8],row[9],row[10],row[11])))
    # user_id,business_id,user_review_count,user_average_stars,business_average_star,business_review_count, actual_star
    #print(test.leftOuterJoin(business_feature).collect())
    test = test.leftOuterJoin(business_feature).map(lambda row:removeNA2(row)).map(lambda row:(row[1][0][0],row[0],row[1][0][1],row[1][0][2],row[1][0][3],row[1][0][4],row[1][0][5],row[1][0][6],row[1][0][7],row[1][0][8],row[1][0][9],row[1][0][10],row[1][1][0],row[1][1][1],row[1][1][2],row[1][1][3],row[1][1][4],row[1][1][5],row[1][1][6],row[1][1][7],row[1][1][8],row[1][1][9],row[1][1][10]))
    test=test.map(lambda row:add_feature(row))




    train_x=train.map(lambda row:[row[2],row[3],row[4],row[5],row[6],row[7],row[8],row[9],row[10],row[11],row[12],row[13],row[14],row[15],row[16],row[17],row[18],row[19],row[20],row[21],row[22],row[24],row[25]]).collect()
    train_y=train.map(lambda row:row[23]).collect()
    test_x=test.map(lambda row:[row[2],row[3],row[4],row[5],row[6],row[7],row[8],row[9],row[10],row[11],row[12],row[13],row[14],row[15],row[16],row[17],row[18],row[19],row[20],row[21],row[22],row[23],row[24]]).collect()
    train_x_df = pd.DataFrame(train_x, columns=['user_review_count', 'user_average_stars', 'fans', 'useful', 'funny', 'cool', 'friend_count', 'compliment_hot', 'compliment_note', 'compliment_photos', 'business_average_star', 'business_review_count' ,'business_open', 'reservation',' credit_card','longitude','latitude','price_range','if_resturant','if_hotel','if_shopping','tip','photo'])
    test_x_df = pd.DataFrame(test_x, columns=['user_review_count', 'user_average_stars', 'fans', 'useful', 'funny', 'cool', 'friend_count', 'compliment_hot', 'compliment_note', 'compliment_photos', 'business_average_star', 'business_review_count' ,'business_open', 'reservation',' credit_card','longitude','latitude','price_range','if_resturant','if_hotel','if_shopping','tip','photo'])
    """train_x_df['SVD_r']=SVD_train
    test_x_df['SVD_r']=SVD_test"""

    """xgb_model = xgb.XGBRegressor(n_estimators=500, max_depth=8)
    xgb_model.fit(train_x, train_y)
    y_pred = xgb_model.predict(test_x)"""

    train_pool = cb.Pool(train_x, label=train_y)


    # Define the model
    params = {
        'loss_function': 'RMSE',
        'iterations': 3500,
        'learning_rate': 0.05,
        'depth': 6,
        'eval_metric': 'RMSE'
    }
    model = cb.CatBoostRegressor(**params)

    # Train the model
    model.fit(train_pool)

    # Evaluate the model
    y_pred = model.predict(test_x)
    test_pair = test.map(lambda row: (row[0], row[1])).collect()
    result_dict = {}
    for i in range(len(test_pair)):
        result_dict[test_pair[i]] = y_pred[i]

    result = yelp_test.map(lambda row: [row[0], row[1], result_dict[(row[0], row[1])]])

    title_csv = ["user_id", "business_id", "prediction"]
    with open(output_file, 'w') as f:
        # using csv.writer method from CSV package
        write = csv.writer(f)
        write.writerow(title_csv)
        write.writerows(result.collect())

    end=time.time()
    print("Duration:{0:.2f} ".format(end - start))