import numpy as np
import pandas as pd

def common_cols():
    return ['Tweet', 'Label']

def models():
    return ['cnn', 'lstm', 'fasttext', 'pretrained ft']

def df_cols():
    return common_cols()+models()

def load_all():
    pred_cnn_l4 = np.load("data/cnn_predict_4.pkl", allow_pickle=True)
    pred_lstm_l4 = np.load("data/lstm_4.pickle", allow_pickle=True)
    pred_fasttext_l4 = np.load("data/fasttext_predict_4.pkl", allow_pickle=True)
    pred_fasttext_pretrained_l4 = np.load("data/fasttext_pre_predict_4.pkl", allow_pickle=True)

    pred_cnn_l3 = np.load("data/cnn_predict_3.pkl", allow_pickle=True)
    pred_lstm_l3 = np.load("data/lstm_3.pickle", allow_pickle=True)
    pred_fasttext_l3 = np.load("data/fasttext_predict_3.pkl", allow_pickle=True)
    pred_fasttext_pretrained_l3 = np.load("data/fasttext_pre_predict_3.pkl", allow_pickle=True)
    
    return pred_cnn_l4,pred_lstm_l4,pred_fasttext_l4,pred_fasttext_pretrained_l4,pred_cnn_l3,pred_lstm_l3,pred_fasttext_l3,pred_fasttext_pretrained_l3

def format_(arr):
    return arr.iloc[:, 0:3]

def merge_all():
    pred_cnn_l4,pred_lstm_l4,pred_fasttext_l4,pred_fasttext_pretrained_l4,pred_cnn_l3,pred_lstm_l3,pred_fasttext_l3,pred_fasttext_pretrained_l3 = load_all()
    
    pred_cnn_l4 = format_(pred_cnn_l4)
    pred_lstm_l4 = format_(pred_lstm_l4)
    pred_fasttext_l4 = format_(pred_fasttext_l4)
    pred_fasttext_pretrained_l4 = format_(pred_fasttext_pretrained_l4)
    
    pred_cnn_l3 = format_(pred_cnn_l3)
    pred_lstm_l3 = format_(pred_lstm_l3)
    pred_fasttext_l3 = format_(pred_fasttext_l3)
    pred_fasttext_pretrained_l3 = format_(pred_fasttext_pretrained_l3)

    tuple_l4=(pred_cnn_l4, pred_lstm_l4.iloc[:, -1:], pred_fasttext_l4.iloc[:, -1:], pred_fasttext_pretrained_l4.iloc[:, -1:])
    pred_l4 = np.concatenate(tuple_l4, axis=1)
    
    tuple_l3=(pred_cnn_l3, pred_lstm_l3.iloc[:, -1:], pred_fasttext_l3.iloc[:, -1:], pred_fasttext_pretrained_l3.iloc[:, -1:])
    pred_l3 = np.concatenate(tuple_l3, axis=1)
    
    df_l4 = pd.DataFrame(data=pred_l4, columns=df_cols())
    df_l3 = pd.DataFrame(data=pred_l3, columns=df_cols())
    return df_l4, df_l3

def label_ordered_models():
    return ['cnn (4 labels)','lstm (4 labels)','fasttext (4 labels)','pretrained ft (4 labels)','cnn (3 labels)','lstm (3 labels)','fasttext (3 labels)','pretrained_ft (3 labels)']

def get_label_counts():
    return [4,4,4,4,3,3,3,3]

def confusion_matrix_preprocessed():
    result=[]
    models = load_all()
    names = label_ordered_models()
    label_counts = get_label_counts()
    
    for i in range(len(models)):
        result.append((models[i].iloc[:,1:3], names[i], label_counts[i]))
    return result