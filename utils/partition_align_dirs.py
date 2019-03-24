"""
This script automatically partitions the align folder evenly.
"""

import os
import shutil

target_dir = 'align'
partition = 100

if __name__ == '__main__':
    targets = os.listdir(target_dir)

    new_dirs = [target_dir + 'subset-{}'.format(i) for i in range(partition)]

    print("Partitioning {} targets in {} into {} partitions".format(len(targets), target_dir, partition))

    cluster_size = int(len(targets) / partition + 1)

    for i in range(partition):
        sub_targets = targets[i * cluster_size : i * cluster_size + cluster_size]

        if os.path.exists(new_dirs[i]):
            shutil.rmtree(new_dirs[i])

        for sub_target in sub_targets:
            shutil.copytree(os.path.join(target_dir, sub_target),os.path.join(new_dirs[i], sub_target))

        print("finish making {} partition".format(i))

