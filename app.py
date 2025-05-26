###############################
#  LRAD サポートチャットボット  #
#   (Perplexity API 版)       #
###############################
"""
必要ファイル / 構成:
────────────────────────────
📄 lrad_perplexity_app.py   ← このファイル (Streamlit アプリ本体)
📄 faq.csv                  ← 質問,回答 の2列のみで構成した CSV
📄 requirements.txt         ← 下記ライブラリを列挙

requirements.txt の例
────────────────────────────
streamlit
pandas
requests
rapidfuzz>=3.0

Secrets の設定 (Streamlit Cloud)
────────────────────────────
[PerplexityAPI]
api_key = "pk-XXXXXXXXXXXXXXXXXXXXXXXXXXXX"

ファイルを配置し `streamlit run lrad_perplexity_app.py` で起動してください。
"""

import streamlit as st
import pandas as pd
import requests
import datetime
from rapidfuzz import fuzz
import unicodedata
import re
import numpy as np

########################
# アプリ基本設定
########################
st.set_page_config(
    page_title="LRADサポートチャット (Perplexity版)",
    page_icon="📘",
    layout="centered"
)

########################
# 定数 / ヘルパ
########################
SYSTEM_PROMPT = (
    "あなたはLRAD（エルラド）という遠赤外線電子熱分解装置の専門家です。"
    "FAQや参考資料を活用しながら、装置の使用方法・注意事項について200文字以内で正確かつ親切に回答してください。"
    "装置に無関係な質問には答えず、関係ない場合は丁寧に断ってください。"
)

API_KEY = st.secrets["PerplexityAPI"]["api_key"]
API_URL = "https://api.perplexity.ai/chat/completions"
MODEL_ID = "llama-3-sonar-small-32k-chat"  # 適宜変更可

########################
# ユーティリティ関数
########################

def is_valid_input(text: str) -> bool:
    """簡易バリデーション (長さ・異常文字)"""
    text = text.strip()
    if len(text) < 3 or len(text) > 300:
        return False
    non_alpha_ratio = len(re.findall(r"[^A-Za-z0-9ぁ-んァ-ヶ一-龠\s]", text)) / len(text)
    if non_alpha_ratio > 0.3:
        return False
    try:
        unicodedata.normalize('NFKC', text).encode('utf-8')
    except UnicodeError:
        return False
    return True

########################
# FAQ ロード (キャッシュ)
########################
@st.cache_data(show_spinner="FAQを読み込み中…")
def load_faq(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # 必須列チェック
    if not set(["質問", "回答"]).issubset(df.columns):
        st.error("faq.csv には '質問' と '回答' 列が必要です。")
        st.stop()
    return df

faq_df = load_faq("faq.csv")

########################
# 類似質問検索 (RapidFuzz)
########################

def search_similar_question(query: str, top_k: int = 1):
    """RapidFuzz の token_set_ratio で最も高い質問を返す"""
    scores = faq_df['質問'].apply(lambda q: fuzz.token_set_ratio(q, query))
    best_idx = int(np.argmax(scores))
    best_score = scores.iloc[best_idx]
    if best_score < 50:  # 閾値以下はマッチ無し扱い
        return None, None, 0
    return faq_df.iloc[best_idx]['質問'], faq_df.iloc[best_idx]['回答'], best_score

########################
# Perplexity への問い合わせ
########################

def call_perplexity(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    body = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }
    try:
        res = requests.post(API_URL, headers=headers, json=body, timeout=45)
        res.raise_for_status()
        data = res.json()
        return data["output"][0]["content"][0]["text"]
    except Exception as e:
        st.error(f"Perplexity API エラー: {e}")
        return "回答を生成できませんでした。時間をおいて再度お試しください。"

########################
# チャットログ保存
########################

def save_log(log):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"chatlog_{ts}.csv"
    pd.DataFrame(log, columns=["質問", "回答"]).to_csv(filename, index=False)
    return filename

########################
# UI レイアウト
########################

st.title("LRAD サポートチャット (Perplexity版)")
st.caption("※FAQとPerplexity AIを用いて回答を生成します。")

if 'chat_log' not in st.session_state:
    st.session_state.chat_log = []

with st.sidebar:
    st.header("設定")
    if st.button("チャットログを保存"):
        fname = save_log(st.session_state.chat_log)
        st.success(f"チャットログを保存しました: {fname}")
        with open(fname, "rb") as f:
            st.download_button("ダウンロード", data=f, file_name=fname, mime="text/csv")

########################
# チャット入力 & 応答
########################

user_input = st.chat_input("質問を入力してください")

if user_input:
    if not is_valid_input(user_input):
        st.error("入力内容に問題があります。3〜300文字で、特殊文字を含めすぎないようにしてください。")
    else:
        # 類似FAQ検索
        ref_q, ref_a, score = search_similar_question(user_input)
        if ref_q is not None:
            reference_block = (
                f"参考FAQ:\n質問: {ref_q}\n回答: {ref_a}\n---\n"
            )
        else:
            reference_block = ""

        # Perplexity へのプロンプト
        prompt = reference_block + f"ユーザー質問: {user_input}"
        answer = call_perplexity(prompt)

        # ログに保存
        st.session_state.chat_log.insert(0, (user_input, answer))

# チャット履歴表示 (新しい順)
for q, a in st.session_state.chat_log:
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        st.markdown(a)
