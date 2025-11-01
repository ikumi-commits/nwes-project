import os
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv
#from google import genai


# .envファイルをロードして環境変数を設定
load_dotenv()

# APIキーを環境変数から取得
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("APIキーが設定されていません。Google CloudのAPIキーを設定してください。")
    st.stop()

# Gemini Client を作成
# client = genai.Client(api_key=api_key)
# Gemini APIを設定
genai.configure(api_key=api_key)


# CSVファイルを読み込む関数を実装してください。
@st.cache_data

def load_data(csv_file_path):
    df = pd.read_csv(csv_file_path)
    return df

# TF-IDFモデルを構築する関数を実装してください。
@st.cache_resource
def build_tfidf_model(texts):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)
    tfidf_matrix = vectorizer.fit_transform(texts)
    return tfidf_matrix, vectorizer

# SentenceTransformerの埋め込みモデルを取得する関数を実装してください。
# SentenceTransformer=文字をベクトル化（word2vecは単語ベース、SentenceTransformerは文章ベース、違いは何か要確認）
@st.cache_resource
def get_embedding_model():
    return SentenceTransformer("sonoisa/sentence-bert-base-ja-mean-tokens")


# テキストデータをベクトル化する関数を実装してください。
@st.cache_resource
def build_embedding_model(texts):
    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings


# ハイブリッド検索を行う関数を実装してください。（意味と単語の両方を加味した検索）
def hybrid_search(query, tfidf_matrix, tfidf_vectorizer, embeddings):
    
    #単語の類似度
    #ユーザの質問をベクトル化
    tfidf_query = tfidf_vectorizer.transform([query])

    #質問と記事の内容の一致度を計算（内積）、スコアが大きいほど質問と単語が一致している記事
    tfidf_scores = (tfidf_matrix @ tfidf_query.T).toarray().ravel()
    
    # 意味の類似度
    # SentenceTransformer類似度
    embed_model = get_embedding_model()
    query_embed = embed_model.encode([query])
    embed_scores = np.dot(embeddings, query_embed.T).ravel()

    # 両方を重み付きで加重平均（0.5ずつ）
    scores = 0.5 * tfidf_scores + 0.5 * embed_scores
    return np.argsort(scores)[::-1]  # 類似度が高い順に並べる


# チャット履歴を初期化する関数を実装してください。
def init_chat_history():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


# チャット履歴を表示する関数を実装してください。
def display_chat_history():
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])



# Geminiモデルを使って応答を生成する関数を実装してください。
# RAG部分
def respond_with_gemini(query, results, texts, top_n=3):

    #類似度が高いn件の記事を呼び出し、１つの文章にまとめる
    context = "\n\n".join([texts[i] for i in results[:top_n]])

    #プロンプトの生成
    prompt = f"以下のニュース記事を参考に質問に答えてください。\n\n質問: {query}\n\n関連記事:\n{context}"
    
    #回答を生成（Gemeniで回答を生成）
    #response = model.generate_content(prompt)

    # Geminiモデルの初期化
    model = genai.GenerativeModel("gemini-pro")
 

    # コンテンツ生成
    response = model.generate_content(prompt)

    # 生成AIで回答
    #response = genai.generate_content(
    #    model="gemini-2.5-flash",
    #    contents=prompt
    #)
    #model = genai.GenerativeModel("gemini-1.5-flash")  # 最新モデル名に注意
    #response = model.generate_content(prompt)

    # 生成された文章を返す
    return response.text 

# Streamlitアプリのメイン
st.title("news RAG System")

# 必要なデータをロードし、処理するコードを実装してください。
csv_file_path = "yahoo_news_articles_preprocessed.csv"
df = load_data(csv_file_path)
texts = df["text_tokenized"].tolist() # 適切なデータを抽出してリストに変換してください。

# TF-IDFモデルを構築してください。
tfidf_matrix, tfidf_vectorizer = build_tfidf_model(texts)  

# 埋め込みモデルを構築してください。
embeddings = build_embedding_model(texts)

init_chat_history()
display_chat_history()

# タブの作成
#tab_chat, tab_history = st.tabs(["チャット", "履歴"])

#with tab_chat:
user_input = st.chat_input("質問を入力してください")

if user_input:

# ユーザの画面に質問内容を表示
# Streamlit で with は「どのUI要素の中に描画するか」を明示するために使われる
    with st.chat_message("user"):
        st.markdown(user_input)

    # 質問内容を履歴に表示・保存
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # ハイブリット検索の実行（質問と記事の類似度を計算、類似度の高い記事から回答を生成する準備）
    results = hybrid_search(user_input, tfidf_matrix, tfidf_vectorizer, embeddings)

    # Gemeniで回答を生成
    answer = respond_with_gemini(user_input, results, texts, top_n=5)

    # 回答をユーザ画面に表示
    with st.chat_message("assistant"):
        st.markdown(answer)

    # 質問内容を履歴に表示・保存
    st.session_state.chat_history.append({"role": "assistant", "content": answer})