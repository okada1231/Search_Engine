import MeCab
import pandas as pd
import numpy as np
import torch
import fugashi
from transformers import BertModel
from transformers import BertJapaneseTokenizer
import re
import copy
import streamlit as st

# パスを取得するのに必要なライブラリをインポート
import subprocess

tagger = MeCab.Tagger()
tagger

# 辞書(mecab-ipadic-NEologd)のPathを取得
cmd=' git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git'
cmd='echo yes | mecab-ipadic-neologd/bin/install-mecab-ipadic-neologd -n -a '
cmd='brew link --overwrite mecab'
cmd='echo `mecab-config --dicdir`"/mecab-ipadic-neologd"'
path = (subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]).decode('utf-8')

# MeCabの事前設定（辞書ファイルをオプションで指定）
tagger = MeCab.Tagger("-d {0}".format(path))

# 分かち書き用tokenizer
tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')

# BERTの日本語学習済みパラメータのモデルです
model = BertModel.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
# print(model) 

# 東北大学_日本語版の設定を確認
from transformers import BertConfig
config_japanese = BertConfig.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')

def st_display_table(df: pd.DataFrame):

    # データフレームを表示
    st.subheader('データの確認')
    st.table(df)

def result():
    search = st.session_state.search
    st.write('検索結果')
    if search == '':
        st.write("結果なし")
    
    else:
        target_layer = -1
        input_ids_kw = tokenizer.encode(search, return_tensors='pt')
        layers = model(input_ids_kw)
        layer = layers[0]
        word_vec_kw = layer[0][target_layer]

        data_list = copy.deepcopy(st.session_state.dl)
        text_list = copy.deepcopy(st.session_state.tl)
        word_vec_list = copy.copy(st.session_state.wv)

        # 文章同士のコサイン類似度を求める
        cos = torch.nn.CosineSimilarity(dim=0)
        cos_sim_list = []
        for i in range(len(data_list)):
            cos_sim = cos(word_vec_kw, word_vec_list[i])
            cos_sim_list.append(cos_sim)

        cos_sim_list = list(map(float, cos_sim_list))
        result = list(zip(cos_sim_list, text_list))
        sort_result = sorted(result, reverse=True)

        for cos_sim_list, text_list in sort_result:
          st.write('類似度: ' + str(cos_sim_list))
          st.write('文章: ' + text_list)

def main():
    st.title("検索システム（仮)")
    activities = ["データ確認", "検索画面"]
    choice = st.sidebar.selectbox("Select Activity", activities)

    if choice == "データ確認":

        uploaded_file = st.sidebar.file_uploader("訓練用データのアップロード", type='csv')

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

                # 表示用テキストリスト
                text_list = []
                for index, data in df_data.iterrows():
                    text_list.append(data['回答'])

                target_layer = -1

                word_vec_list =  []
                for i in range(len(data_list)):
                    input_ids = tokenizer.encode(data_list[i], return_tensors='pt')  # ptはPyTorchの略
                    layers = model(input_ids)
                    layer = layers[0]
                    word_vec = layer[0][target_layer]
                    word_vec_list.append(word_vec)

                # データフレームをセッションステートに退避（名称:df）
                st.session_state.df = copy.deepcopy(df)
                st.session_state.dl = copy.deepcopy(data_list)
                st.session_state.tl = copy.deepcopy(text_list)
                st.session_state.wv = copy.copy(word_vec_list)

                # スライダーの表示（表示件数）
                cnt = st.sidebar.slider('表示する件数', 1, len(df), 10)

                # テーブルの表示
                st_display_table(df.head(int(cnt)))

        else:
            st.subheader('訓練用データをアップロードしてください')
        
    if choice == "検索画面":

                # セッションステートにデータフレームがあるかを確認
            if 'df' in st.session_state:

                # セッションステートに退避していたデータフレームを復元
                df = copy.deepcopy(st.session_state.df)

                st.text_input("検索入力欄", key="search")
                st.caption("入力例（資金調達　方法）（起業　資金）")
                if st.button("検索"):
                    result()
                
            else:
                st.subheader('訓練用データをアップロードしてください')

if __name__ == "__main__":
    main()
