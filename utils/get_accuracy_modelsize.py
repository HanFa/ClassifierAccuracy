import os
from itertools import product

model_types = ['SVC']
data_sizes = [1]

prediction_dir = '/Users/fang/Desktop/FaceEmbeddingClassifiers/predictions/' # Change this to the directory of the prediction results
measure_model_size = False # change this to true to measure the model sizes



def get_accuracy(l1, l2):
    assert len(l1) == len(l2)
    return float(sum([x == y for (x, y) in zip(l1, l2)])) / len(l1)

if __name__ == '__main__':

    # Load in groud-truth label
    labels = []
    with open(os.path.join(prediction_dir, 'labels.csv')) as f:
        lines = f.readlines()
        for line in lines:
            labels.append(int(line.strip().split(',')[0]))


    # Get accuracy for each senarios
    if os.path.exists('results-accuracy.tsv'):
        os.remove('results-accuracy.tsv')

    if os.path.exists('results-modelsize.tsv'):
        os.remove('results-modelsize.tsv')

    for model in model_types:
        open('results-accuracy.tsv', 'a').write(model + '\t')
        if measure_model_size:
            open('results-modelsize.tsv', 'a').write(model + '\t')

        for data_size in data_sizes:
            fname = os.path.join(prediction_dir, 'predict_{}-{}.csv'.format(model, data_size))
            predictions = []
            with open(fname, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    predictions.append(int(line.strip()))

            accuracy = get_accuracy(labels, predictions)
            open('results-accuracy.tsv', 'a').write('{}\t'.format(accuracy))
            if measure_model_size:
                open('results-modelsize.tsv', 'a').write(
                    '{}\t'.format(
                        os.path.getsize('../dataset/embeddings-{}/{}.pkl'.format(data_size, model))))
            
        open('results-accuracy.tsv', 'a').write('\n')
        open('results-modelsize.tsv', 'a').write('\n')
