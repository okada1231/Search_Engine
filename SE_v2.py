import MeCab
import pandas as pd
import numpy as np
import torch
import fugashi
from transformers import BertModel
from transformers import BertJapaneseTokenizer
from transformers import BertConfig
import re
import string
import copy
import streamlit as st

from transformers import logging
logging.set_verbosity_error()

# 分かち書き用tokenizer
tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')

# BERTの日本語学習済みパラメータのモデルです
model = BertModel.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
# print(model) 

# 東北大学_日本語版の設定を確認
config_japanese = BertConfig.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')

#　類似度の取得
def result():
    search = st.session_state.search
    st.write('検索結果')
    if search == '':
        #　何も入力されていない場合に表示
        st.write("結果なし")
    
    else:
        input_ids_kw = tokenizer.encode(search, return_tensors='pt')
        layers = model(input_ids_kw)
        target_layer = -1
        layer = layers[0]
        word_vec_kw = layer[0][target_layer]
        
        data_list = copy.deepcopy(st.session_state.dl)
        text_list = copy.deepcopy(st.session_state.tl)
        word_vec_list = copy.copy(st.session_state.wv)
        

        # 文章同士のコサイン類似度を求める
#         cos = torch.nn.CosineSimilarity(dim=0)
#         cos_sim_list = []
#         for i in range(len(data_list)):
#             cos_sim = cos(word_vec_kw, word_vec_list[i])
#             cos_sim_list.append(cos_sim)

#         cos_sim_list = list(map(float, cos_sim_list))
#         result = list(zip(cos_sim_list, text_list))
#         sort_result = sorted(result, reverse=True)

#         for cos_sim_list, text_list in sort_result:
#           st.write('類似度: ' + str(cos_sim_list))
#           st.write('文章: ' + text_list)

def main():
    st.title("検索システム（仮)")

    uploaded_file = st.sidebar.file_uploader("データのアップロード", type='csv')


    if uploaded_file is not None:

        ufile = copy.deepcopy(uploaded_file)

        try:
            # 文字列の判定
            pd.read_csv(ufile, encoding="utf_8_sig")
            enc = "utf_8_sig"
        except:
            enc = "shift-jis"

        finally:
            # データフレームの読み込み
            df = pd.read_csv(uploaded_file, encoding=enc) 
            pd.set_option("display.max_colwidth", 500)

            df_data = df

            # 比較用データリスト
            data_list = []
            for index, data in df_data.iterrows():
                data_list.append(data['質問事項'] + data['回答'])
            
            # データリストの不要な文字を置き換え
            for i in range(len(data_list)):
                data_list[i] = data_list[i].replace('\r','')
                data_list[i] = data_list[i].replace('\n','')
                data_list[i] = data_list[i].translate(str.maketrans('','', string.punctuation))
                data_list[i] = data_list[i].replace('、','')
                data_list[i] = data_list[i].replace('・','')
                data_list[i] = data_list[i].replace('。','')

            # 表示用テキストリスト
            text_list = []
            for index, data in df_data.iterrows():
                text_list.append(data['回答'])
            
            word_vec_list =  []
            for i in range(len(data_list)):
                input_ids = tokenizer.encode(data_list[i], return_tensors='pt')  # ptはPyTorchの略
                layers = model(input_ids)
                target_layer = -1
                layer = layers[0]
                word_vec = layer[0][target_layer]
                word_vec_list.append(word_vec)

            # データフレームをセッションステートに退避
            
            st.session_state.df = copy.deepcopy(df)
            st.session_state.dl = copy.deepcopy(data_list)
            st.session_state.tl = copy.deepcopy(text_list)
            st.session_state.wv = copy.copy(word_vec_list)

            st.text_input("検索入力欄", key="search")
            st.caption("入力例（資金調達　方法）（起業　資金）")
            if st.button("検索"):
                result()

    else:
        st.subheader('データをアップロードしてください')
        
                

if __name__ == "__main__":
    main()
