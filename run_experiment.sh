#!/bin/bash

function align_faces {
    ./utils/align-dlib.py --dlibFacePredictor ./dlib/shape_predictor_68_face_landmarks.dat ./dataset/raw align \
    outerEyesAndNose ./dataset/align --size 96 
}

# Generate representation for each dataset
function generate_rep() {
    ./batch-represent/main.lua -model ./openface/nn4.small2.v1.t7 -outDir dataset/embeddings -data dataset/align
}

# Split representations for train and test set
function split_embeddings {
    python utils/split_embeddings.py
}

# Fit and dump the classifiers
function fit_classifiers {
    for i in 1 2 3 4 5 10 25 50 100 200
    do
        python classifier.py --dlibFacePredictor dlib/shape_predictor_68_face_landmarks.dat --imgDim 96 \
        --networkModel openface/nn4.small2.v1.t7 train dataset/embeddings-$i --classifier linearSVC
    done
}

# Predict 
function predict_validation {
    for i in 1 2 3 4 5 10 25 50 100 200
    do
        python classifier.py --dlibFacePredictor dlib/shape_predictor_68_face_landmarks.dat --imgDim 96 \
        --networkModel openface/nn4.small2.v1.t7 \
        infer --classifierModel ./dataset/embeddings-$i/classifier.pkl --embeddings_dir ./dataset/validation --result_name linearSVC-$i
    done
}

split_embeddings
fit_classifiers
predict_validation