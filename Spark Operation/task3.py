import pyspark,sys, json,time,operator


if __name__ == '__main__':
    # input_r_path=sys.argv[1]
    # input_b_path=sys.argv[2]
    # output_a_path=sys.argv[3]
    # output_b_path=sys.argv[4]
    input_r_path = './review.json'
    input_b_path = './business.json'
    output_a_path = './out-task3a.txt'
    output_b_path = './out-task3b.json'

    sc = pyspark.SparkContext('local[*]', 'task3')
    sc.setLogLevel("WARN")

    start=time.time()
    review_rdd = sc.textFile(input_r_path).map(lambda row: json.loads(row)).map(lambda row:(row['business_id'],row['stars']))
    business_rdd=sc.textFile(input_b_path).map(lambda row: json.loads(row)).map(lambda row:(row['business_id'],row['city']))
    combined_rdd=review_rdd.leftOuterJoin(business_rdd)
    result=combined_rdd.map(lambda row:(row[1][1],row[1][0],1)).filter(lambda row:row[0] is not None )\
          .flatMap(lambda x:[(x[0],(x[1],x[2]))]).reduceByKey(lambda a,b:(a[0]+b[0],a[1]+b[1]))\
          .map(lambda row:(row[0],row[1][0]/row[1][1]))
    mid=time.time()
    result_s10=result.takeOrdered(10, key=lambda row: [-row[1],row[0]])
    print(result_s10)
    end_s=time.time()
    result_p=result.collect()
    result_p=sorted(result_p,key=lambda row: (-row[1],row[0]))[:10]
    print(result_p)
    end_p=time.time()
    result_3b={}
    result_3b['m1']=(mid-start)+(end_p-end_s)
    result_3b['m2']=end_s-start
    result_3b['reason']='I think this happen because spark use map reduce function to make it run faster.'
    result_s = result.sortBy(lambda row: [-row[1], row[0]]).collect()
    with open(output_a_path, 'w') as f:
        f.write('city,stars \n')
        for row in result_s:
            f.write(row[0])
            f.write(',')
            f.write(str(row[1]))
            f.write('\n')

    with open(output_b_path, 'w') as o:
        json.dump(result_3b, o, sort_keys=False)

