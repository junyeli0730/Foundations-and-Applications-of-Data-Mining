import sys,pyspark,time,csv
import binascii
import statistics
from blackbox import BlackBox
num_hash=200
bit_array_length=69997
a_list=[1, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997, 1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069, 1087, 1091, 1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151, 1153, 1163, 1171, 1181, 1187, 1193, 1201, 1213, 1217, 1223]
b_list=[79826, 86379, 93902, 32099, 30313, 52863, 14293, 51536, 94517, 8096, 22447, 38020, 74075, 73591, 97640, 21722, 43345, 31282, 23685, 61305, 43589, 78381, 9793, 99030, 64643, 47001, 90695, 7403, 3019, 44717, 96931, 15943, 53607, 59244, 86929, 55813, 45074, 68906, 82123, 20669, 65564, 57242, 23342, 53172, 22473, 22632, 50886, 50903, 34523, 44274, 8462, 49341, 21366, 15668, 84949, 1376, 76592, 44470, 21624, 1187, 36047, 2438, 61343, 1271, 96870, 79933, 86062, 23908, 18897, 68622, 24746, 75220, 13494, 62465, 15939, 69443, 88138, 7829, 53357, 46609, 23019, 94741, 73300, 36500, 87284, 80597, 56217, 15712, 14336, 3157, 3147, 485, 61657, 28703, 33012, 90536, 55605, 52324, 24085, 4010, 57793, 67606, 3590, 28742, 37267, 12496, 43815, 33138, 89684, 50711, 74431, 39379, 40931, 58041, 52215, 30736, 69411, 17028, 39977, 59837, 72814, 44166, 30457, 53380, 59759, 51804, 73482, 50340, 14064, 16027, 58024, 90957, 90705, 94064, 96689, 88194, 55667, 67886, 93572, 28429, 28910, 73554, 41857, 89430, 76036, 1621, 72022, 51646, 10194, 60056, 85141, 25751, 15144, 72752, 44315, 71991, 56518, 16786, 68522, 82321, 44096, 85989, 36500, 12561, 52624, 26756, 57749, 65680, 77752, 63701, 36835, 87765, 77492, 69046, 52937, 49572, 63220, 76642, 89558, 61743, 33743, 40237, 82914, 17781, 56175, 12190, 92707, 85782, 53792, 82976, 81067, 21143, 45102, 1802, 81171, 14048, 40917, 57462, 34026, 60568]
group_size=8
def myhashs(s):
    result=[]
    for k in range(num_hash):
        r=((a_list[k]*s)+b_list[k])%bit_array_length
        result.append(r)
    return result
if __name__ == '__main__':
    start=time.time()
    file_name='users.txt'
    stream_size=300
    num_of_asks=50
    output_filenames='out2.csv'

    """file_name=sys.argv[1]
    stream_size =int(sys.argv[2])
    num_of_asks =int(sys.argv[3])
    output_filenames =sys.argv[4]"""

    bx=BlackBox()
    estimation=[]
    for _ in range(num_of_asks):
        longest_trailing_zero = [0] * num_hash
        stream_users=bx.ask(file_name,stream_size)
        for user in stream_users:
            user=int(binascii.hexlify(user.encode('utf8')),16)
            hash_list=myhashs(user)
            trailing_zeros=list(map(lambda v:len(bin(v))-len(bin(v).rstrip("0")),hash_list))
            longest_trailing_zero=list(map(lambda x,y: max(x,y),longest_trailing_zero,trailing_zeros))
        expected_value=list(map(lambda r: 2**r,longest_trailing_zero))
        avg_hash_value=[]

        for v in range(0,num_hash,group_size):
            avg=sum(expected_value[v:v+4])/group_size
            avg_hash_value.append(avg)
        estimation.append(int(statistics.median(avg_hash_value)))
    print(sum(estimation)/(stream_size*num_of_asks))
    time_list=list(range(num_of_asks))
    ground_truth_list=[stream_size]*num_of_asks
    with open(output_filenames, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Time','Ground Truth','Estimation'])
        writer.writerows(zip(time_list,ground_truth_list,estimation))

    end = time.time()
    print("Duration:{0:.2f} ".format(end - start))