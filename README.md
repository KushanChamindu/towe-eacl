# TOWE-EACL

Code for EACL 2021 paper : Attention-based Relational Graph Convolutional Network for Target-Oriented Opinion Words Extraction

Models and results can be found at our EACL 2021 paper https://www.aclweb.org/anthology/2021.eacl-main.170.pdf

#### requirement
+ python==3.7.7
+ numpy==1.19.4
+ pandas==1.1.4
+ torch==1.7.0
+ torch-cluster==1.5.8
+ torch-scatter==2.0.5
+ torch-sparse==0.6.8
+ torch-spline-conv==1.2.0
+ torch-geometric==1.6.1
+ tqdm==4.46.0
+ fitlog==0.9.13
+ spacy==2.3.4
+ transformers==4.1.1

#### pre-requisites

1. prepare spacy model for dependency parse.
    * we prepare en_core_web_sm-2.2.5.tar.gz at ./data/spacy_model, please unzip it.
2. create folders ./models , ./log , ./logs in the root directory of this project.
    * ./models is the folder where model store.
    * ./log is the folder stored log recording train and valid process.
    * ./logs is the folder stored logs which fitlog generate.
3. prepare bert model.
    * please download https://huggingface.co/bert-base-uncased

#### model

* Target-BiLSTM
* ARGCN

#### train and evaluate

bash run.sh

#### Newly added features
* Created a new Student Feedback Data set and can find that through https://github.com/KushanChamindu/Student_feedback_analysis_dataset
* Added Capsule Network Model. https://github.com/KushanChamindu/Capsule_network_text_pytorch

#### Publication
[Dataset and Baseline for Automatic Student Feedback Analysis](https://aclanthology.org/2022.lrec-1.219) (Herath et al., LREC 2022)
