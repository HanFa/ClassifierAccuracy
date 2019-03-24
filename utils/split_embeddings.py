import os
from collections import Counter

dataset_embedding_dir = '../dataset/embeddings/' # directory of the face embeddings of the dataset
output_dir = '../dataset/' # output directory of partitioned face embeddings

if __name__ == '__main__':

    labels = []
    with open(os.path.join(dataset_embedding_dir, 'labels.csv')) as f:
        lines = f.readlines()
        for line in lines:
            label = line.strip().split(',')[0]
            labels.append(label)

    counter = Counter(labels)
    filtered_identity = list(map(lambda x: x[0], filter(lambda x: x[1] > 205, counter.items())))

    identity_to_reps = dict.fromkeys(filtered_identity, None)

    with open(os.path.join(dataset_embedding_dir, 'reps.csv')) as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            label = str(labels[idx])
            rep = line.strip()

            if label in identity_to_reps.keys():
                if identity_to_reps[label]:
                    identity_to_reps[label].append(rep)
                else:
                    identity_to_reps[label] = [ rep ]


    train_datasizes = [1, 2, 3, 4, 5, 10, 25, 50, 100, 200]
    valid_per_identity = 5
    smallest_datasize = len(sorted(identity_to_reps.items(), key=lambda x: len(x[1]))[0][1])
    assert max(train_datasizes) + valid_per_identity < smallest_datasize

    # prepare training sets
    for train_datasize in train_datasizes:
        os.mkdir(os.path.join(output_dir, 'embeddings-{}'.format(train_datasize)))
        for identity, reps in identity_to_reps.items():
            open(os.path.join(output_dir, 'embeddings-{}/labels.csv'.format(train_datasize), 'a')).write('{},dataset/align/n000001/0424_01.png\n'.format(identity) * train_datasize)
            open(os.path.join(output_dir, 'embeddings-{}/reps.csv'.format(train_datasize), 'a')).write('\n'.join(reps[:train_datasize]) + '\n')

    # prepare validation set
    os.mkdir(os.path.join(output_dir,'validation'))
    for identity, reps in identity_to_reps.items():
        open(os.path.join(output_dir, 'validation/labels.csv', 'a')).write('{},dataset/align/n000001/0424_01.png\n'.format(identity) * valid_per_identity)
        open(os.path.join(output_dir, 'validation/reps.csv', 'a')).write('\n'.join(reps[-valid_per_identity:]) + '\n')

    





