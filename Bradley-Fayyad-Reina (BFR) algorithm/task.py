
import sys,time,pyspark
import numpy as np
from sklearn.cluster import KMeans
import math


if __name__ == '__main__':
    start=time.time()
    input_file='hw6_clustering.txt'
    n_cluster=10
    output_file='output.txt'
    final_res={}
    """input_file =sys.argv[1]
    n_cluster =int(sys.argv[2])
    output_file =sys.argv[3]"""
    data_np = np.loadtxt(input_file, delimiter=",")
    total_point=data_np.shape[0]
    #Step 1. Load 20% of the data randomly.
    train_size = int(data_np.shape[0] * 0.2)
    indices = np.random.permutation(data_np.shape[0])
    training_idx, test_idx = indices[:train_size], indices[train_size:]
    training, remaining = data_np[training_idx, :], data_np[test_idx, :]
    train_X = training[:,2:]
    n_dimension=train_X.shape[1]
    #Step 2. Run K-Means (e.g., from sklearn) with a large K (e.g., 5 times of the number of the input clusters)km = KMeans(
    km1=KMeans(n_clusters=5*n_cluster).fit_predict(train_X)
    #Step 3. In the K-Means result from Step 2, move all the clusters that contain only one point to RS (outliers).
    v, c = np.unique(km1, return_counts=True)
    RS_cluster_index=list(np.argwhere(c == 1))
    RS_index=[]
    for c_i in RS_cluster_index:
        RS_index.append(np.where(km1==c_i[0])[0][0])
    RS=train_X[RS_index,:]
    RS_all=training[RS_index,:]
    if len(RS_index)!=0:
        RS_index.sort(reverse=True)
        for row in RS_index:
            train_X=np.delete(train_X,row,0)
            training=np.delete(training, row, 0)
    #Step 4. Run K-Means again to cluster the rest of the data points with K = the number of input clusters.
    km2= KMeans(n_clusters=n_cluster).fit_predict(train_X)
    #Step 5. Use the K-Means result from Step 4 to generate the DS clusters (i.e., discard their points and generate statistics).
    DS=np.concatenate((train_X,km2.reshape(-1,1)),1)
    r1=np.concatenate((training,km2.reshape(-1,1)),1)
    for p in r1:
        final_res[int(p[0])]=p[-1]
    sc = pyspark.SparkContext('local[*]', 'task1')
    sc.setLogLevel("WARN")
    def cal_DS_stats(rows):
        SUM=[0]*n_dimension
        SUMSQ=[0]*n_dimension
        for row in rows:
            for d in range(n_dimension):
                SUM[d]+=row[d]
                SUMSQ+=row[d]*row[d]
        return [len(rows),SUM,SUMSQ]
    DS_rdd = sc.parallelize(DS).map(lambda row:[row[-1],row[:-1]]).groupByKey().mapValues(cal_DS_stats)
    DS_dic=DS_rdd.collectAsMap()

    #Step 6. Run K-Means on the points in the RS with a large K (e.g., 5 times of the number of the input clusters) to generate CS (clusters with more than one points) and RS (clusters with only one point).
    k_3=int(1+RS.shape[0]/2)
    cs_size = k_3
    km3=KMeans(n_clusters=k_3, random_state=0, n_init=10).fit_predict(RS)
    v, c = np.unique(km3, return_counts=True)
    RS_cluster_index = list(np.argwhere(c == 1))
    RS_index = []
    for c_i in RS_cluster_index:
        RS_index.append(np.where(km3 == c_i[0])[0][0])
    CS_dict={}
    CS_summarized={}
    RS_new=[]
    for k in range(len(RS)):
        if k in RS_index:
            RS_new.append(RS_all[k])
        else:
            if km3[k] in CS_dict:
                CS_dict[km3[k]].append(RS_all[k])
                CS_summarized[km3[k]]['COUNT'] += 1
                for d in range(n_dimension):
                    CS_summarized[km3[k]]['SUM'][d]+=RS[k][d]
                    CS_summarized[km3[k]]['SUMSQ'][d] += RS[k][d] ** 2
            else:
                CS_dict[km3[k]]=[]
                CS_dict[km3[k]].append(RS_all[k])
                CS_summarized[km3[k]]={}
                CS_summarized[km3[k]]['COUNT']=1
                CS_summarized[km3[k]]['SUM']=RS[k]
                CS_summarized[km3[k]]['SUMSQ']=np.zeros(n_dimension)
                for d in range(n_dimension):
                    CS_summarized[km3[k]]['SUMSQ'][d]=RS[k][d]**2
    output=[]
    run='Round 1: '
    discard_points = 0
    for i in range(n_cluster):
        discard_points += DS_dic[i][0]
    n_CS=len(list(CS_summarized.keys()))
    CS_points=0
    for key in CS_summarized:
        CS_points+=CS_summarized[key]['COUNT']
    RS_points=len(RS_new)
    str1=run+str(discard_points)+','+str(n_CS)+','+str(CS_points)+','+str(RS_points)
    output.append(str1)
    for r in range(4):
        #Step 7. Load another 20% of the data randomly.
        if r==3:
            training=remaining
        else:
            indices = np.random.permutation(remaining.shape[0])
            training_idx, test_idx = indices[:train_size], indices[train_size:]
            training, remaining = remaining[training_idx, :], remaining[test_idx, :]
        train_X = training[:, 2:]
        #Step 8. For the new points, compare them to each of the DS using the Mahalanobis Distance and assign them to the nearest DS clusters if the distance is<2 𝑑.
        criteria=2*math.sqrt(n_dimension)
        for point in training:
            point1=point[2:]
            min_d = 10000000
            for c in range(n_cluster):
                centroid = DS_dic[c]
                res = 0
                num_points = centroid[0]
                SUM = centroid[1]
                SUMSQ = centroid[2]
                SDV = []
                centroid_avg = []
                for i in range(len(SUM)):
                    SDV.append(math.sqrt(SUMSQ[i] / num_points - ((SUM[i] / num_points) ** 2)))
                    centroid_avg.append(SUM[i] / num_points)
                for i in range(len(point1)):
                    res += pow((point1[i] - centroid_avg[i]) / SDV[i], 2)
                if math.sqrt(res) < min_d:
                    min_d = math.sqrt(res)
                    min_i = c
            if min_d < criteria:
                final_res[int(point[0])]=min_i
                DS_dic[min_i][0] += 1
                for i in range(len(point1)):
                    DS_dic[min_i][1][i] += point1[i]
                    DS_dic[min_i][2][i] += point1[i] ** 2
            else:
                #For the new points that are not assigned to DS clusters, using the Mahalanobis Distance and assign the points to the nearest CS clusters if the distance is<2 𝑑
                min_d = 10000000
                for key in CS_summarized:
                    res = 0
                    num_points = CS_summarized[key]['COUNT']
                    SUM = CS_summarized[key]['SUM']
                    SUMSQ = CS_summarized[key]['SUMSQ']
                    SDV = []
                    centroid_avg = []
                    for i in range(len(SUM)):
                        SDV.append(math.sqrt(SUMSQ[i] / num_points - ((SUM[i] / num_points) ** 2)))
                        centroid_avg.append(SUM[i] / num_points)
                    for i in range(len(point1)):
                        res += pow((point1[i] - centroid_avg[i]) / SDV[i], 2)
                    if math.sqrt(res) < min_d:
                        min_d = math.sqrt(res)
                        min_i = key
                if min_d < criteria:
                    CS_dict[min_i].append(point)
                    CS_summarized[min_i]['COUNT'] += 1
                    for i in range(len(point1)):
                        CS_summarized[min_i]['SUM'][i] += point1[i]
                        CS_summarized[min_i]['SUMSQ'][i] += point1[i] ** 2
                else:
                    #Step 10. For the new points that are not assigned to a DS cluster or a CS cluster, assign them to RS.
                    RS_new.append(point)
        #Step 11. Run K-Means on the RS with a large K (e.g., 5 times of the number of the input clusters) to generate CS (clusters with more than one points) and RS (clusters with only one point).
        RS = np.array([np.array(xi) for xi in RS_new])
        RS_feature=RS[:,2:]
        k4=int(1+RS.shape[0]/2)
        km4 = KMeans(n_clusters=k4, random_state=0, n_init=10).fit_predict(RS_feature)
        km4=cs_size+km4

        v, c = np.unique(km4, return_counts=True)
        RS_cluster_index = list(np.argwhere(c == 1))

        RS_index = []
        for c_i in RS_cluster_index:
            RS_index.append(np.where(km4 == (c_i[0]+cs_size))[0][0])
        RS_new = []
        for k in range(RS.shape[0]):
            if k in RS_index:
                RS_new.append(RS[k])
            else:
                if km4[k] in CS_dict:
                    CS_dict[km4[k]].append(RS[k])
                    CS_summarized[km4[k]]['COUNT'] += 1
                    for d in range(n_dimension):
                        CS_summarized[km4[k]]['SUM'][d] += RS_feature[k][d]
                        CS_summarized[km4[k]]['SUMSQ'][d] += RS_feature[k][d] ** 2
                else:
                    CS_dict[km4[k]] = []
                    CS_dict[km4[k]].append(RS[k])
                    CS_summarized[km4[k]] = {}
                    CS_summarized[km4[k]]['COUNT'] = 1
                    CS_summarized[km4[k]]['SUM'] = RS_feature[k]
                    CS_summarized[km4[k]]['SUMSQ'] = np.zeros(n_dimension)
                    for d in range(n_dimension):
                        CS_summarized[km4[k]]['SUMSQ'][d] = RS_feature[k][d] ** 2
        cs_size += k4
        #Step12.Merge CS clusters that have a Mahalanobis Distance <2 𝑑.
        key_list=list(CS_summarized.keys())
        i=0
        while len(key_list)>1 and i<len(key_list):
            key1=key_list[i]
            for j in range(i+1,len(key_list)):
                key2=key_list[j]
                sum1=CS_summarized[key1]
                sum2=CS_summarized[key2]
                avg1=sum1['SUM']/sum1['COUNT']
                avg2=sum2['SUM']/sum2['COUNT']
                sdv2=sum2['SUMSQ']/sum2['COUNT']-avg2**2
                dis=0
                for d in range(n_dimension):
                    dis+=((avg1[d]-avg2[d])/sdv2[d])**2
                dis=math.sqrt(dis)
                if dis<criteria:
                    CS_dict[key1]+=CS_dict[key2]
                    CS_summarized[key1]['COUNT']+=CS_summarized[key2]['COUNT']
                    CS_summarized[key1]['SUM']=np.add(CS_summarized[key1]['SUM'],CS_summarized[key2]['SUM'])
                    #CS_summarized[key1]['SUM']=[ CS_summarized[key1]['SUM'][d]+CS_summarized[key2]['SUM'][d] for d in range(n_dimension)]
                    CS_summarized[key1]['SUMSQ']=np.add(CS_summarized[key1]['SUMSQ'],CS_summarized[key2]['SUMSQ'])
                    #CS_summarized[key1]['SUMSQ'] = [CS_summarized[key1]['SUMSQ'][d] + CS_summarized[key2]['SUMSQ'][d] for d in range(n_dimension)]
                    CS_dict.pop(key2)
                    CS_summarized.pop(key2)
            key_list=list(CS_summarized.keys())
            i+=1

        print(CS_dict[key][0][0])
        if r==3:

            for key in CS_summarized:
                sum1=CS_summarized[key]
                avg1=sum1['SUM']/sum1['COUNT']
                min_dis=1000000
                for c in range(n_cluster):
                    num_points=DS_dic[c][0]
                    SUM=DS_dic[c][1]
                    SUMSQ=DS_dic[c][2]
                    SDV=[]
                    centroid_avg=[]
                    for i in range(len(SUM)):
                        SDV.append(math.sqrt(SUMSQ[i] / num_points - ((SUM[i] / num_points) ** 2)))
                        centroid_avg.append(SUM[i] / num_points)
                m_dis=0
                for d in range(n_dimension):
                    m_dis+=((avg1[d]-centroid_avg[d])/SDV[d])**2
                m_dis=math.sqrt(m_dis)
                if m_dis<min_dis:
                    min_dis=m_dis
                    min_c=c
            if min_dis<criteria:
                DS_dic[c][0]+=sum1['COUNT']
                for d in range(n_dimension):
                    DS_dic[c][1][d]+=sum1['SUM'][d]
                    DS_dic[c][2][d] +=sum1['SUMSQ'][d]
                CS_summarized.pop(key)
                for cs_p in range(sum1['COUNT']):
                    print(CS_dict[key])
                    index=CS_dict[key][cs_p][0]
                    final_res[int(index)]=min_c


        run = 'Round '+str(2+r)+': '
        discard_points =0
        for i in range(n_cluster):
            discard_points+=DS_dic[i][0]
        n_CS = len(list(CS_summarized.keys()))
        CS_points = 0
        for key in CS_summarized:
            CS_points += CS_summarized[key]['COUNT']
        RS_points = len(RS_new)
        str1 = run + str(discard_points) + ',' + str(n_CS) + ',' + str(CS_points) + ',' + str(RS_points)
        output.append(str1)
        print(output)


    file = open(output_file, 'w')
    file.write('The intermediate results:\n')
    for w in output:
        file.write(w+'\n')
    file.write('The clustering results:\n')
    for i in range(total_point):
        if i in final_res:
            file.write(str(i)+','+str(int(final_res[i]))+'\n')
        else:
            file.write(str(i) + ',-1' + '\n')
    file.close()
    end = time.time()
    print("Duration:{0:.2f} ".format(end - start))

