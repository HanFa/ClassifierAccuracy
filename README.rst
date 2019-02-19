Openface Classifier Accuracy vs. Data Size
============================================

.. contents::


Research Objectives
-----------------------------
* Study how well the different types of classifiers classify face embeddings, when given a different size of labeled face images
* Study how the size of embeddings and classifier change as the dataset increasing

Hypothesis
-----------------------------
* Accuracy of classification will increase until a saturation points as the data size increases
* The cross entropy loss will decrease as the data size increases

Experiment Details
---------------------

Dataset
~~~~~~~~~~
We use a **subset** of `VGGFace2 <http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/>`_ which consists of 500 identities with 100 face images per identity on average. 
Respectively, the classifier is fit with {1, 2, 3, 4, 5, 10, 25, 50, 100} image person. The validation dataset includes 2500 images with 5 images per identity. 

Evaluation
~~~~~~~~~~~~~~~~~~~~
In order to evaluate the classifier, the **accuracy** and **cross entropy loss** is calculated based on the 2500-image validation set. Suppose that

* :math:`\hat y` is the predictions
* :math:`y` is the one-hot ground truth

.. math::

    \text{Accuracy} = \text{# Correct Preds} / 2500 \\
    L(\hat y, y) = -y \log(\hat y) + (1-y) \log(1 - \hat y)

Steps
~~~~~~~~~~~~~~~~
To reproduce the results, please refer to `run_experiment.sh <run_experiment.sh>`_. The first step is to align faces in the dataset.

.. code::

    # Preprocess: Align the faces for the dataset: ./data/images/raw
    function align_faces {
        ./util/align-dlib.py --dlibFacePredictor ./dlib/shape_predictor_68_face_landmarks.dat ./dataset/raw align \
        outerEyesAndNose ./dataset/align --size 96 
    }

The second step is to extract embeddings from those aligned images.

.. code:: 

    # Generate representation for each dataset
    function generate_rep() {
        ./batch-represent/main.lua -model ./openface/nn4.small2.v1.t7 -outDir dataset/embeddings -data models/align
    }

The third step is to partition the embeddings. For each identity, split the embeddings into {1, 2, 3, 4, 5, 10, 25, 50, 100, 200} embeddings per identity. Also, it leaves out 5 images for each identity as the validation set.

.. code:: 

    # Split representations for train and test set
    function split_embeddings {
        python3 split_embeddings.py
    }

The fourth step is to fit classifiers with the training set.



Results
-----------


