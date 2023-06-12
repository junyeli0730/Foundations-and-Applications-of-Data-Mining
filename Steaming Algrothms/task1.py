import sys,pyspark,time,csv
import binascii

from blackbox import BlackBox
bit_array_length=69997
a_list=[1,3,5,7,9,11,13,17,19,23,29,31,37,41,43,47, 53, 59, 61, 67]
b_list=[4215, 476, 2105, 4559, 5562, 2586, 8531, 8536, 2396, 5611, 3367, 656, 7749, 6469, 8620, 2114, 479, 580, 4580, 6009]
bit_array=[0]*69997
def myhashs(s):
    result=[]
    for k in range(5):
        r=((a_list[k]*s)+b_list[k])%bit_array_length
        result.append(r)
    return result
if __name__ == '__main__':
    start=time.time()
    file_name='users.txt'
    stream_size=100
    num_of_asks=30
    output_filenames='out1.csv'

    """file_name=sys.argv[1]
    stream_size =int(sys.argv[2])
    num_of_asks =int(sys.argv[3])
    output_filenames =sys.argv[4]"""
    fpr_list=[]
    predict_list=[]
    visited_list=[]
    bx=BlackBox()
    for _ in range(num_of_asks):
        stream_users=bx.ask(file_name,stream_size)
        """sc = pyspark.SparkContext('local[*]', 'task1')
        sc.setLogLevel("WARN")
        user_data=sc.textFile(file_name)"""
        for user in stream_users:
            #check if it has been added before, a=0 no,a=1 yes
            if user in visited_list:
                a=1
            else:
                a=0
            visited_list.append(user)
            user_int=int(binascii.hexlify(user.encode('utf8')),16)
            hash_value=myhashs(user_int)
            p=0
            for v in hash_value:
                if bit_array[v]==1:
                    p=1
                bit_array[v]=1
            if p==1 and a==0:
                predict_list.append(1)
            else:
                predict_list.append(0)
        fpr_list.append(sum(predict_list)/len(predict_list))
    print(fpr_list)
    time_list=list(range(num_of_asks))


    with open(output_filenames, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Time','FPR'])
        writer.writerows(zip(time_list, fpr_list))
    end = time.time()
    print("Duration:{0:.2f} ".format(end - start))