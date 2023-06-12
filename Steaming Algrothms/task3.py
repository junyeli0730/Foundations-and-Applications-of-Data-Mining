import sys,pyspark,time,csv
import binascii
import statistics
from blackbox import BlackBox
import random
sample_size=100
seq_num=0
sample_list=[]
if __name__ == '__main__':
    random.seed(553)
    start=time.time()
    file_name='users.txt'
    stream_size=100
    num_of_asks=30
    output_filenames='out3.csv'

    """file_name=sys.argv[1]
    stream_size =int(sys.argv[2])
    num_of_asks =int(sys.argv[3])
    output_filenames =sys.argv[4]"""

    bx=BlackBox()
    res=[]
    for _ in range(num_of_asks):
        stream_users = bx.ask(file_name, stream_size)
        for user in stream_users:
            seq_num += 1
            if len(sample_list)<sample_size:
                sample_list.append(user)
            else:
                prob_discard=random.random()
                if prob_discard<sample_size/seq_num:
                    #we accept the sample
                    index_to_discard=random.randint(0,99)
                    sample_list[index_to_discard]=user
            if seq_num%100==0:
                r=[seq_num,sample_list[0],sample_list[20],sample_list[40],sample_list[60],sample_list[80]]
                res.append(r)
    time_list=list(range(num_of_asks))

    with open(output_filenames, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['seqnum','0_id','20_id','40_id','60_id','80_id'])
        writer.writerows(res)

    end = time.time()
    print("Duration:{0:.2f} ".format(end - start))