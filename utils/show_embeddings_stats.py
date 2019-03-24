import os
from collections import Counter


if __name__ == '__main__':

    labels = []
    
    with open('../dataset/embeddings/labels.csv') as f:
        lines = f.readlines()
        for line in lines:
            label = line.strip().split(',')[0]
            labels.append(label)

    # labels_nodup = list(set(labels))

    # with open('dataset/embeddings/labels-altered.csv', 'w') as f:
    #     lines = open('dataset/embeddings/labels.csv').readlines()
    #     for line in lines:
    #         _, img_path = line.strip().split(',')
    #         label = img_path.split('/')[-2]
    #         f.write(','.join([str(labels_nodup.index(label)), img_path]) + '\n')

    counter = dict(Counter(labels))
    counter = sorted(counter.items(), key=lambda x: x[1], reverse=True)

    total_num = len(counter)

    for thres in [105, 205, 255]:
        left_num = len(list(filter(lambda x: x[1] > thres, counter)))
        print("{} out of {} identities have face images more than {}"
            .format(left_num, total_num, thres))    

