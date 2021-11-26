# AIHUB 2021 한국어 멀티모달 음성인식 데이터 경진대회

# MNC 코드 템플릿

## Directory 
 '''

    \base_builder : model build directory
        - model_builder.py 
    \base_model : model directory
        - base_model.py
    \checkpoint : model save / load 관련
        - checkpoint.py
    \data_folder : 배포한 data
        - Train, Test, Noise
    \dataloader : data loader 관련
        - augment.py Specaugment code
        - data_loader.py data_loader
        - feature.py Audio Feature Mel filter bank / MFCC 등등
        - vocabulary.py Vocabulary token 관련
    \dataset : dataset 관련
        - labels.csv Tokenization 매핑
        - Train.txt Train dataset 매핑
        - Test.txt Test dataset 매핑
    \metric : metric 관련
        - metric.py : CER, WER 등
        - wer_utils.py
    \outputs : record directory train results (model, Tensorboard)
    train.py : train script
    train.yaml : train config
    preprocessing.py : Video, Audio, Label, Token 매핑
    train.py : train script
    vid2npy.py : Video save to numpy in data_folder/Train(Test)/Video_npy
    README.md
 '''
