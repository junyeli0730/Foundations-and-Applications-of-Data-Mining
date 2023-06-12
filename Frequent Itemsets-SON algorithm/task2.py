# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import pyspark,sys,time,csv,json,math
def construct1(list_b,s):
    candidate={}
    frequent_result=set()
    can=[]
    for baseket_set in list_b:
        for item in baseket_set:
            if item in candidate:
                candidate[item]+=1
            else:
                candidate[item]=1
    for key in candidate:
        can_set=set()
        can_set.add(key)
        can.append(can_set)
        if candidate[key]>s:
            frequent_result.add(frozenset({key}))
    return can,frequent_result

def construct2(frequent,size_of_c):
    candidates=set()
    for itema in frequent:
        for itemb in frequent:
            union_set=itema.union(itemb)
            if len(union_set)==size_of_c:
                candidates.add(union_set)
    return candidates


def find_frequent(candidates,list_b,s):
    frequent_res=set()
    frequent_d={}
    for basket in list_b:
        for c in candidates:
            check = all(item in basket for item in c)
            if check==True:
                if c in frequent_d:
                    frequent_d[c]+=1
                else:
                    frequent_d[c]=1
    for key in frequent_d:
        if frequent_d[key]>s:
            frequent_res.add(key)
    return frequent_res
def convert_fs_to_tuple(fs):
    result=set()
    for i in fs:
        result.add(tuple(i))
    return result

def apriori(basket,support):
    #only need to return frequent item set
    frequent_itemset=set()
    candidate1, frequent1 = construct1(basket, support)
    frequent_itemset=convert_fs_to_tuple(frequent1)
    size_of_c = 2
    candidates = construct2(frequent1, size_of_c)
    while len(candidates) != 0:
        frequent_item = find_frequent(candidates, basket, support)
        frequent_itemset=frequent_itemset.union(convert_fs_to_tuple(frequent_item))
        size_of_c += 1
        candidates = construct2(frequent_item, size_of_c)
    return frequent_itemset

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    #filter_threshold=int(sys.argv[1])
    #support=int(sys.argv[2])
    #input_path=sys.argv[3]
    #output_path=sys.argv[4]
    filter_threshold=20
    support=9
    input_path="./ta_feng_all_months_merged.csv"
    output_path='./out2.txt'

    sc = pyspark.SparkContext('local[*]', 'task2')
    sc.setLogLevel("WARN")
    #read data
    test_rdd=sc.textFile(input_path).filter(lambda row: not row.startswith('"TRANSACTION_DT"'))\
            .map(lambda row: row.split(',')).map(lambda row:(row[0].replace('"',''),int(row[1].replace('"','')),int(row[5].replace('"',''))))
    test_rdd=test_rdd.map(lambda row: [row[0]+'-'+str(row[1]),row[2]])
    title_csv=["DATE-CUSTOMER_ID", "PRODUCT_ID"]

    with open('customer_product.csv', 'w') as f:
        # using csv.writer method from CSV package
        write = csv.writer(f)
        write.writerow(title_csv)
        write.writerows(test_rdd.collect())
    start = time.time()
    customer_product=sc.textFile('./customer_product.csv').filter(lambda row: not row.startswith("DATE-CUSTOMER_ID"))\
    .map(lambda row:row.split(','))
    basket=customer_product.groupByKey().mapValues(set).filter(lambda row:len(row[1])>filter_threshold) \
        .map(lambda row: row[1])

    basket_size=basket.count()

    def SON1_map(set_row):
        basket_l=list(set_row)
        ps=math.ceil((len(basket_l)/basket_size)*support)
        return apriori(basket_l,ps)
    SON_candidates=basket.mapPartitions(SON1_map).distinct().sortBy(lambda row:[len(row),row])
    #print(SON_candidates.collect())
    SON_candidates_l=SON_candidates.collect()
    def SON2_map(set_rows):
        result_d={}
        basket_l=list(set_rows)
        for row in basket_l:
            for son_candidate in SON_candidates_l:
                if all(item in row for item in son_candidate):
                    if son_candidate in result_d:
                        result_d[son_candidate]+=1
                    else:
                        result_d[son_candidate]=1
        res_son2=set()
        for pair in result_d:
            res_son2.add(tuple([pair,result_d[pair]]))
        return res_son2
    SON2_candidates = basket.mapPartitions(SON2_map)
    SON_Frequent_itemsets=SON2_candidates.reduceByKey(lambda a,b:a+b).filter(lambda row:row[1]>support).map(lambda row:row[0])\
    .sortBy(lambda row:[len(row),row])
    #print(SON_Frequent_itemsets.collect())
    SON_Frequent_itemsets_l=SON_Frequent_itemsets.collect()
    size_c=0
    size_f=0
    with open(output_path, 'a') as o:
        o.write("Candidates:")
        for tup in SON_candidates_l:
            if len(tup)==size_c:
                o.write(',(')
            else:
                o.write('\n'+'(')
                size_c+=1
            for t in range(size_c):
                if (t!=size_c-1):
                    o.write("'"+tup[t]+"', ")
                else:
                    o.write("'"+tup[t]+"'")
            o.write(')')
        o.write('\n')
        o.write('Frequent Itemsets:')
        for tup in SON_Frequent_itemsets_l:
            if len(tup) == size_f:
                o.write(',(')
            else:
                o.write('\n' + '(')
                size_f += 1
            for t in range(size_f):
                if (t != size_f - 1):
                    o.write("'" + tup[t] + "', ")
                else:
                    o.write("'" + tup[t] + "'")
            o.write(')')
    o.close()

    end=time.time()
    print("Duration:{0:.2f} ".format(end-start))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
filter_threshold=int(sys.argv[1])
    support=int(sys.argv[2])
    input_path=sys.argv[3]
    output_path=sys.argv[4]
    """
    filter_threshold=20
    support=9
    input_path="./ta_feng_all_months_merged.csv"
    output_path='./out2.txt'
    """
    sc = pyspark.SparkContext('local[*]', 'task2')
    sc.setLogLevel("WARN")
    #read data
    test_rdd=sc.textFile(input_path).filter(lambda row: not row.startswith('"TRANSACTION_DT"'))\
            .map(lambda row: row.split(',')).map(lambda row:(row[0].replace('"',''),int(row[1].replace('"','')),int(row[5].replace('"',''))))
    test_rdd=test_rdd.map(lambda row: [row[0]+'-'+str(row[1]),row[2]])
    title_csv=["DATE-CUSTOMER_ID", "PRODUCT_ID"]

    with open('customer_product.csv', 'w') as f:
        # using csv.writer method from CSV package
        write = csv.writer(f)
        write.writerow(title_csv)
        write.writerows(test_rdd.collect())
    start = time.time()
    customer_product=sc.textFile('./customer_product.csv').filter(lambda row: not row.startswith("DATE-CUSTOMER_ID"))\
    .map(lambda row:row.split(','))
    basket=customer_product.groupByKey().mapValues(set).filter(lambda row:len(row[1])>filter_threshold) \
        .map(lambda row: row[1])