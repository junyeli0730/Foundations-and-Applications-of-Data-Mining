import pyspark, sys, time, csv, math


def avg(x):
    list(x)
    return sum(x) / len(x)


if __name__ == '__main__':
    start = time.time()
    """
    input_train_file = sys.argv[1]
    input_test_file=sys.argv[2]
    output_file= sys.argv[3]

    """

    input_train_file = './yelp_train.csv'
    input_test_file = './test.csv'
    output_file = './output2_1.csv'

    """
    input_train_file = './train.csv'
    input_test_file = './test.csv'
    output_file = './output2_1.csv'
    """
    sc = pyspark.SparkContext('local[*]', 'task2_1')
    sc.setLogLevel("WARN")
    yelp_train = sc.textFile(input_train_file).filter(lambda row: not row.startswith("user_id")) \
        .map(lambda row: row.split(',')).map(lambda row: [row[0], row[1], float(row[2])])
    yelp_test = sc.textFile(input_test_file).filter(lambda row: not row.startswith("user_id")) \
        .map(lambda row: row.split(' , '))

    """yelp_pair_rate = yelp_train.map(lambda row: [(row[0], row[1]), row[2]]).collectAsMap()
    user_rdd = yelp_train.map(lambda row: (row[0], row[2])).groupByKey().mapValues(
        lambda row: (len(row), avg(row))).sortBy(lambda row: -row[1][0])
    user_top100 = user_rdd.map(lambda row: (row[0], row[1][1])).take(100)
    business_rdd = yelp_train.map(lambda row: (row[1], row[2])).groupByKey().mapValues(
        lambda row: (len(row), avg(row))).sortBy(lambda row: -row[1][0])
    business_top100 = business_rdd.map(lambda row: (row[0], row[1][1])).take(100)
    imputation=[]
    for i in range(100):
        for j in range(100):
            if (user_top100[i][0],business_top100[j][0]) not in yelp_pair_rate:
                #print((user_top100[i][0],business_top100[j][0]))
                avgValue=[user_top100[i][0],business_top100[j][0],(user_top100[i][1]+business_top100[j][1])/2]
                imputation.append(avgValue)
    yelp_train_list=yelp_train.collect()
    imp_train=yelp_train_list+imputation
    yelp_train=sc.parallelize(imp_train)
    print(len(imputation))"""

    yelp_u_b = yelp_train.map(lambda row: (row[0], row[1])).groupByKey().mapValues(list).collectAsMap()
    yelp_b_u = yelp_train.map(lambda row: (row[1], row[0])).groupByKey().mapValues(set).collectAsMap()
    yelp_pair_rate = yelp_train.map(lambda row: [(row[0], row[1]), row[2]]).collectAsMap()
    avg_all = yelp_train.map(lambda row: row[2]).mean()
    avg_rating_item = yelp_train.map(lambda row: (row[1], float(row[2]))).groupByKey().mapValues(avg).collectAsMap()
    avg_rating_user = yelp_train.map(lambda row: (row[0], float(row[2]))).groupByKey().mapValues(avg).collectAsMap()
    #print(yelp_train.map(lambda row: (row[0], row[1])).groupByKey().mapValues(len).mean(key=lambda k:k[1]))
    """
    def compute_similarity(b1, b2):
        u1 = yelp_b_u[b1]
        u2 = yelp_b_u[b2]
        return len(u1.intersection(u2)) / len(u1.union(u2))


    c = yelp_test.filter(lambda row: row[1] in yelp_b_u)
    c1 = c.map(lambda row: [(row[0], row[1]), yelp_u_b[row[0]]])
    c1 = c1.flatMap(lambda row: [(row[0], row[1][i]) for i in range(len(row[1]))])

    similarity = c1.map(lambda row: [row[0], (row[1], compute_similarity(row[0][1], row[1]))])


    def ax(x):
        return max(list(x), key=lambda item: item[1])


    sim_dict = similarity.groupByKey().mapValues(ax).map(lambda row: [row[0], row[1][0]]).collectAsMap()


    """
    weight_dict = {}


    def weight_calculation(b1, b2):
        u1 = yelp_b_u[b1]
        u2 = yelp_b_u[b2]
        n = 0
        d1 = 0
        d2 = 0
        u = list(u1.intersection(u2))
        #if no common user
        if len(u)==0:
            mean_diff=abs(avg_rating_item[b1]-avg_rating_item[b2])
            if mean_diff<=0.85:
                return 1
            elif mean_diff<=2:
                return 0.85
            elif mean_diff<=3:
                return 0.5
            elif mean_diff<=4:
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
        #print(len(u))
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
                #w = w * pow(abs(w), (2.5 - 1))
                weight_dict[pair] = w
            else:
                w = weight_dict[pair]
            if w>0:
                weight_list.append((b_l[i], w))
        print(len(weight_list))
        if len(weight_list)==0:
            return (avg_rating_item[business] + avg_rating_user[user]) / 2
        weight_list.sort(key=lambda item: item[1], reverse=True)
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
        """if p <= 1:
            p = (avg_rating_item[business] + avg_rating_user[user]) / 2"""
            #p = avg_rating_user[user]
        return p

    yelp_predict = yelp_test.map(lambda row: (row[0], row[1], IB_predict(row[0], row[1])))

    title_csv = ["user_id", "business_id", "prediction"]

    with open(output_file, 'w') as f:
        # using csv.writer method from CSV package
        write = csv.writer(f)
        write.writerow(title_csv)
        write.writerows(yelp_predict.collect())

    end = time.time()
    print("Duration:{0:.2f} ".format(end - start))

