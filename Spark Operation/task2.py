import pyspark,sys, json, time
def p_func(x):
    return hash(x[0])

if __name__ == '__main__':
    # input_path=sys.argv[1]
    # output_path=sys.argv[2]
    #n_partition=sys.argv[3]
    input_path='./review.json'
    output_path='./out-task2.json'
    n_partition=2
    result={}
    result['default']={}
    result['customized']={}

    sc=pyspark.SparkContext('local[*]','task2')
    sc.setLogLevel("WARN")
    #default
    review_rdd = sc.textFile(input_path).map(lambda row: json.loads(row)).map(lambda row: (row["business_id"],1))
    result['default']['n_partition']=review_rdd.getNumPartitions()
    result['default']['n_items']=review_rdd.glom().map(len).collect()
    start_d = time.time()
    result_d=review_rdd.reduceByKey(lambda a,b:a+b).takeOrdered(10, key=lambda row: [-row[1],row[0]])
    end_d=time.time()
    result['default']['exe_time']=end_d-start_d
    #customize
    review_rdd_c=review_rdd.partitionBy(n_partition,p_func)
    result['customized']['n_partition']=n_partition
    result['customized']['n_items']=review_rdd_c.glom().map(len).collect()
    start_c=time.time()
    result_c=review_rdd_c.reduceByKey(lambda a,b:a+b).takeOrdered(10, key=lambda row: [-row[1],row[0]])
    end_c=time.time()
    result['customized']['exe_time']=end_c-start_c
    with open(output_path, 'w') as o:
        json.dump(result, o, sort_keys=False)
