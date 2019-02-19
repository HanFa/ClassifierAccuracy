
import sys
import pickle

if __name__ == '__main__':
    for i in [1, 2, 3, 4, 5, 10, 25, 50, 100, 200]:
        with open('dataset/embeddings-{}/classifier.pkl'.format(i), 'rb') as f:
            if sys.version_info[0] < 3:
                    (le, clf) = pickle.load(f)
            else:
                    (le, clf) = pickle.load(f, encoding='latin1')

        with open('dataset/embeddings-{}/model.pkl'.format(i), 'w') as f:
            pickle.dump(clf, f)

