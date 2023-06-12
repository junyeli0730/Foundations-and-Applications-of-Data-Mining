import time,pyspark,graphframes,sys
from pyspark.sql import SparkSession
from pyspark.sql import Row
import os

os.environ["PYSPARK_SUBMIT_ARGS"] = "--packages graphframes:graphframes:0.8.2-spark3.1-s_2.12 pyspark-shell"

if __name__ == '__main__':
    start=time.time()

    filter_threshold=7
    input_file='./ub_sample_data.csv'
    output_file='./output1.txt'

    """filter_threshold =int(sys.argv[1])
    input_file =sys.argv[2]
    output_file =sys.argv[3]"""

    sc = pyspark.SparkContext('local[*]', 'task1')
    sc.setLogLevel("WARN")
    data_rdd = sc.textFile(input_file).filter(lambda row: not row.startswith("user_id")) \
        .map(lambda row: row.split(','))
    node=data_rdd.map(lambda row:row[0]).distinct()
    node_list=node.collect()
    user_business_dict=data_rdd.groupByKey().mapValues(set).collectAsMap()
    user_pair=node.flatMap(lambda row:[(row,x) for x in node_list])
    def edge_filter(pair):
        u1=pair[0]
        u2=pair[1]
        if u1==u2:
            return False
        b1=user_business_dict[u1]
        b2=user_business_dict[u2]
        if len(b1.intersection(b2))>=filter_threshold:
            return True
        return False
    edges=user_pair.filter(edge_filter)

    conf = pyspark.SparkConf().setAppName("task1").setMaster("local[*]")
    spark = SparkSession.builder.config(conf=conf).getOrCreate()

    edge_df=spark.createDataFrame(edges, ['src', 'dst'])
    node_rdd = edges.map(lambda row:row[0]).distinct().map(lambda x: Row(id=x))
    node_df = spark.createDataFrame(node_rdd)
    g=graphframes.GraphFrame(node_df,edge_df)
    result = g.labelPropagation(maxIter=5)
    result_rdd=result.rdd
    result_rdd=result_rdd.map(lambda row:[row[1],row[0]]).groupByKey().mapValues(list)
    result_rdd=result_rdd.map(lambda row:sorted(row[1])).sortBy(lambda row:[len(row),row[0]])
    r=result_rdd.collect()
    file = open(output_file, 'w')
    for item in r:
        item_str=str(item)
        item_str=item_str[1:-1]
        file.write(item_str + "\n")
    file.close()
    end=time.time()
    print("Duration:{0:.2f} ".format(end - start))
