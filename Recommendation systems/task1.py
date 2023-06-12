import pyspark, sys, time, csv


def minhash(user, user_n):
    # permutation of index
    a_list = [1, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103,
              107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223,
              227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347,
              349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463,
              467, 479, 487, 491, 499, 503, 509, 521, 523, 541]
    b_list = [4215, 476, 2105, 4559, 5562, 2586, 8531, 8536, 2396, 5611, 3367, 656, 7749, 6469, 8620, 2114, 479, 580,
              4580, 6009, 4466, 7074, 5592, 5213, 8342, 8194, 6563, 2514, 6756, 2963, 8012, 8276, 3077, 4510, 1766,
              1267, 7323, 6618, 3125, 6053, 1341, 5342, 3799, 2366, 2103, 1084, 310, 2464, 8100, 7169, 373, 1510, 2144,
              419, 5811, 4928, 589, 6532, 6005, 5060, 3610, 8000, 8393, 2603, 4234, 292, 1926, 3592, 1078, 3924, 7593,
              1399, 1708, 2733, 7310, 5415, 8145, 3354, 8367, 7962, 4504, 858, 6837, 921, 5188, 6793, 2103, 3746, 6627,
              8641, 888, 1992, 4618, 3846, 1209, 2996, 5796, 6597, 4324, 4864]
    permutation_list = [min(user)]
    for i in range(100):
        p1 = []
        for j in user:
            p1.append((a_list[i] * j + b_list[i]) % user_n)
        permutation_list.append(min(p1))
    return tuple(permutation_list)


def LSH(column):
    hash_n = 100
    b = 50
    r = 2
    c = []
    for i in range(b):
        bucket = []
        for j in range(r):
            bucket.append(column[i * r + j])
        c.append(7 * (hash(tuple(bucket)) + 59) % 10796)
    return tuple(c)


def jaccard_similarity(c1, c2, dic):
    return len(dic[c1].intersection(dic[c2])) / len(dic[c1].union(dic[c2]))


if __name__ == '__main__':
    start = time.time()
    """
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    """
    input_file = './yelp_train.csv'
    output_file = './output1.csv'


    sc = pyspark.SparkContext('local[*]', 'task1')
    sc.setLogLevel("WARN")
    yelp_train = sc.textFile(input_file).filter(lambda row: not row.startswith("user_id")) \
        .map(lambda row: row.split(','))
    raw_rdd = sc.textFile(input_file).filter(lambda row: not row.startswith("user_id")) \
        .map(lambda row: row.split(','))

    # from set to boolean matrices
    user_rdd = yelp_train.map(lambda x: x[0]).distinct().zipWithIndex()
    user_dict = user_rdd.collectAsMap()
    user_n = len(user_dict)
    business_rdd = yelp_train.map(lambda x: x[1]).distinct().zipWithIndex()
    business_dict = business_rdd.collectAsMap()
    business_n = len(business_dict)
    # business, user
    yelp_train = yelp_train.map(lambda row: (row[1], row[0]))
    # business:{user_id1,userid_2}
    yelp_train = yelp_train.map(lambda row: (business_dict[row[0]], user_dict[row[1]])).groupByKey().mapValues(set)
    # minhash
    yelp_minhash = yelp_train.mapValues(lambda user_list: minhash(user_list, user_n))

    # LSH
    yelp_lsh = yelp_minhash.mapValues(lambda sig: LSH(sig))
    yelp_lsh = yelp_lsh.flatMap(lambda row: [((row[1][i], i), row[0]) for i in range(len(row[1]))])

    # find candidate: in same bucket
    candidates = yelp_lsh.groupByKey().mapValues(list)


    def create_pair(l):
        r = []
        for i in range(len(l)):
            for j in range(i + 1, len(l)):
                k = (l[i], l[j])
                r.append(k)
        return r


    candidates_pair = candidates.flatMap(lambda row: create_pair(row[1])).distinct()

    # compute jacard similarity
    yelp_dict = yelp_train.collectAsMap()
    result = candidates_pair.map(
        lambda pair: (pair[0], pair[1], jaccard_similarity(pair[0], pair[1], yelp_dict))).filter(
        lambda row: row[2] >= 0.5)

    business_flip_dict = business_rdd.map(lambda row: (row[1], row[0])).collectAsMap()
    result = result.map(lambda row: [business_flip_dict[row[0]], business_flip_dict[row[1]], row[2]])
    print(result.collect())

    def sort_row(row):
        r = sorted([row[0], row[1]])
        r.append(row[2])
        return r


    result = result.map(lambda row: sort_row(row)).sortBy(lambda x: [x[0],x[1]])

    title_csv = ["business_id_1", " business_id_2", " similarity"]

    with open(output_file, 'w') as f:
        # using csv.writer method from CSV package
        write = csv.writer(f)
        write.writerow(title_csv)
        write.writerows(result.collect())

    end = time.time()
    print("Duration:{0:.2f} ".format(end - start))

