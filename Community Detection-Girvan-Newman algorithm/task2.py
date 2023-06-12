import time,pyspark,sys

class Node:
    def __init__(self, value):
        self.value = value
        self.children = []
        self.parents = []
        self.level=0
        self.indegree=0
        self.credit=0

def BFS(adj_list,node_list,root):
    node_dict={}
    for node in node_list:
        node_dict[node]=Node(node)
        if node!=root:
            node_dict[node].credit=1
    root=node_dict[root]
    level=0
    visited= set()
    visited.add(root.value)
    v=set()
    q = []
    q.append(root)
    while len(q)>0:
        cur=q.pop(0)
        neighbour_list=adj_list[cur.value]
        if level != cur.level:
            visited=visited.union(v)
            v=set()
            level=cur.level
        for n in neighbour_list:
            if n not in visited:
                n = node_dict[n]
                cur.children.append(n)
                n.parents.append(cur)
                n.level=cur.level+1
                n.indegree=n.indegree+1
                if n not in v:
                    q.append(n)
                v.add(n.value)
    return node_dict

def credit_cal(node_dict,edge_credit):
    #find leaves
    leaves=set()
    for node in node_dict:
        if len(node_dict[node].children)==0:
            leaves.add(node_dict[node])
    #assign credit
    leaves=list(leaves)
    while len(leaves)>0:
        cur=leaves.pop(0)
        if len(cur.parents)!=0:
            credit_each_parent=cur.credit/len(cur.parents)
            for p in cur.parents:
                p.credit=p.credit+credit_each_parent
                p.children.remove(cur)
                if len(p.children)==0:
                    leaves.append(p)
                pair=tuple(sorted([cur.value,p.value]))
                edge_credit[pair]+=credit_each_parent
    return edge_credit

def cal_betweenness(adj_list,node_list,edge_dict):
    for r in node_list:
        node_dict=BFS(adj_list,node_list,r)
        """for v in node_dict:
            print(node_dict[v].value,node_dict[v].level,node_dict[v].children)"""
        edge_dict=credit_cal(node_dict,edge_dict)

    for e in edge_dict:
        edge_dict[e]=edge_dict[e]/2
    return edge_dict

def detect_community(adj_list,node_list):
    unvisited=node_list
    res=[]
    while len(unvisited)>0:
        start=unvisited.pop(0)
        visited=set()
        q=[start]
        visited.add(start)
        while len(q)>0:
            cur=q.pop(0)
            neighbour_list=adj_list[cur]
            for n in neighbour_list:
                if n not in visited:
                    unvisited.remove(n)
                    visited.add(n)
                    q.append(n)
        res.append(visited)
    return res
def cal_modularity(adj_list,node_list,edge_list,node_degree_dict,edge_num):
    m2=edge_num*2
    community=detect_community(adj_list,node_list)
    Q=0
    for c in community:
        c_node_list=c
        for n1 in c_node_list:
            for n2 in c_node_list:
                pair=(n1,n2)
                if pair in edge_list:
                    A=1
                else:
                    A=0
                k1=node_degree_dict[n1]
                k2=node_degree_dict[n2]
                Q+=A-((k1*k2)/m2)
    return Q/m2,community


if __name__ == '__main__':
    start=time.time()

    filter_threshold=7
    input_file='./ub_sample_data.csv'
    betweenness_output_file='./output2_b.txt'
    community_output_file='./out2.txt'
    """filter_threshold =int(sys.argv[1])
    input_file =sys.argv[2]
    output_file =sys.argv[3]"""

    sc = pyspark.SparkContext('local[*]', 'task1')
    sc.setLogLevel("WARN")
    data_rdd = sc.textFile(input_file).filter(lambda row: not row.startswith("user_id")).map(lambda row: row.split(','))
    node = data_rdd.map(lambda row: row[0]).distinct()
    node_list = node.collect()
    user_business_dict = data_rdd.groupByKey().mapValues(set).collectAsMap()
    user_pair = node.flatMap(lambda row: [(row, x) for x in node_list])
    def edge_filter(pair):
        u1 = pair[0]
        u2 = pair[1]
        if u1 == u2:
            return False
        b1 = user_business_dict[u1]
        b2 = user_business_dict[u2]
        if len(b1.intersection(b2)) >= filter_threshold:
            return True
        return False
    edges = user_pair.filter(edge_filter)
    raw_edge=user_pair.filter(edge_filter)
    edge_list=edges.collect()
    node_rdd=edges.map(lambda row:row[0]).distinct()
    adj_list=edges.groupByKey().mapValues(list).collectAsMap()
    node_list=node_rdd.collect()
    edges_dictionary=edges.map(lambda row:tuple(sorted(list(row)))).distinct().map(lambda row:(row,0)).collectAsMap()
    #task2.1
    edge_dic=cal_betweenness(adj_list,node_list,edges_dictionary)
    edge_betweeness=edges.map(lambda row:tuple(sorted(list(row)))).distinct().map(lambda row:(row,round(edge_dic[row],5))).sortBy(lambda row:[-row[1],row[0][0]])
    result_b=edge_betweeness.collect()
    file = open(betweenness_output_file, 'w')
    for item in result_b:
        item_str = str(item)
        item_str = item_str[1:-1]
        file.write(item_str + "\n")
    file.close()
    #task2.2
    edge_num=len(edge_dic)
    node_degree_dict=edges.groupByKey().mapValues(len).collectAsMap()
    cur_modularity,cur_community=cal_modularity(adj_list,node_list,edge_list,node_degree_dict,edge_num)
    max_modularity = cur_modularity
    max_community = cur_community
    while(len(edge_list)>0):
        #remove edge
        highest_betweenness=result_b[0][1]
        edge_removed_list=[]
        for edge_b in result_b:
            if edge_b[1]==highest_betweenness:
                edge_removed_list.append(edge_b[0])
            if edge_b[1]<highest_betweenness:
                break
        for edge_removed in edge_removed_list:
            edge_removed_2=(edge_removed[1],edge_removed[0])
            edge_list.remove(edge_removed)
            edge_list.remove(edge_removed_2)
            adj_list[edge_removed[0]].remove(edge_removed[1])
            adj_list[edge_removed[1]].remove(edge_removed[0])
        cur_modularity,cur_community = cal_modularity(adj_list, node_rdd.collect(), edge_list, node_degree_dict, edge_num)
        if cur_modularity>max_modularity:
            max_modularity=cur_modularity
            max_community=cur_community
        #calculate betweenness
        edges_dictionary = sc.parallelize(edge_list).map(lambda row:tuple(sorted(list(row)))).distinct().map(lambda row:(row,0)).collectAsMap()
        node_list=node_rdd.collect()
        edge_dic = cal_betweenness(adj_list, node_list, edges_dictionary)
        edge_betweeness = sc.parallelize(edge_list).map(lambda row: tuple(sorted(list(row)))).distinct().map(lambda row: (row, round(edge_dic[row], 5))).sortBy(lambda row: [-row[1], row[0][0]])
        result_b = edge_betweeness.collect()
    result=sc.parallelize(max_community).map(lambda row: sorted(list(row))).sortBy(lambda row:[len(row),row[0]]).collect()
    file = open(community_output_file, 'w')
    for item in result:
        item_str = str(item)
        item_str = item_str[1:-1]
        file.write(item_str + "\n")
    file.close()
    end=time.time()
    print("Duration:{0:.2f} ".format(end - start))