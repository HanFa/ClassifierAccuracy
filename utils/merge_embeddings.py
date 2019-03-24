import os
import shutil

if __name__ == '__main__':

    total_label_lines = []
    total_rep_lines = []
    
    for i in range(1, 100):
        print("reading in subset #{} ...".format(i))
        label_lines = open('../dataset/embeddings-subset-{}/labels.csv'.format(i), 'r').readlines()
        rep_lines = open('../dataset/embeddings-subset-{}/reps.csv'.format(i), 'r').readlines()

        total_label_lines += label_lines
        total_rep_lines += rep_lines

    target_dir = '../dataset/embeddings'
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)

    os.mkdir(target_dir)

    print('write into the merged ...')
    open(os.path.join(target_dir, 'labels.csv'), 'w').writelines(total_label_lines)
    open(os.path.join(target_dir, 'reps.csv'), 'w').writelines(total_rep_lines)

    