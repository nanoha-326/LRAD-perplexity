# FAQチャットボット with Perplexity補完・類似検索・ログ保存（不正入力対応版）

import streamlit as st
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import datetime
import secret_keys          # 既存のまま利用可（不要なら削除）
from PIL import Image        # 既存のまま利用可（不要なら削除）
import re
import unicodedata
import numpy as np

# --- Streamlitの設定 ---
st.set_page_config(page_title="LRADサポートチャット", page_icon="\U0001F4D8", layout="centered")

# --- 初期設定 ---
# Perplexity APIキー（.streamlit/secrets.toml に [PerplexityAPI] api_key="..." を追加）
PERPLEXITY_API_KEY = st.secrets.PerplexityAPI.api_key

# SentenceTransformer は FAQ のベクトル化で使用
model = SentenceTransformer('all-MiniLM-L6-v2')

system_prompt = """
あなたはLRAD専用のチャットボットです。
「LRAD（エルラド）」とは熱分解装置（遠赤外線電子熱分解装置）のことで、これは有機廃棄物の処理装置です。
あなたの役割は、この装置の検証をサポートすることです。

以下の点を守ってください：
・あなたはLRADの専門家として利用者の質問にわかりやすく回答し、処理検証をサポートできます。
・装置に関連することのみを答えてください。それ以外の質問（例：天気、有名人、趣味、思想、料理、政治、ゲーム、スポーツ、健康など）には絶対に答えないでください。
・世間話をされてもLRADに関係のない場合は答えないでください。
・質問には親切に、できるだけ分かりやすく答えてください。
・FAQのファイル内に類似する情報がない場合は、回答が不明であることを丁寧に伝え、適切に対応してください。
"""

# --- 入力バリデーション関数 ---
def is_valid_input(text: str) -> bool:
    text = text.strip()
    if len(text) < 3 or len(text) > 300:
        return False
    non_alpha_ratio = len(re.findall(r'[^A-Za-z0-9ぁ-んァ-ヶ一-龠\s]', text)) / len(text)
    if non_alpha_ratio > 0.3:
        return False
    try:
        normalized = unicodedata.normalize('NFKC', text)
        normalized.encode('utf-8')
    except UnicodeError:
        return False
    return True

# --- FAQ CSVの読み込み ---
@st.cache_data
def load_faq(csv_file: str) -> pd.DataFrame:
    df = pd.read_csv(csv_file)
    df['embedding'] = df['質問'].apply(lambda x: model.encode(x))
    return df

faq_df = load_faq("faq.csv")

# --- 類似質問検索 ---
def find_similar_question(user_input: str, faq_df: pd.DataFrame):
    user_vec = np.array(model.encode([user_input]))
    faq_vecs = np.stack(faq_df['embedding'].values)
    scores = cosine_similarity(user_vec, faq_vecs)[0]
    top_idx = scores.argmax()
    return faq_df.iloc[top_idx]['質問'], faq_df.iloc[top_idx]['回答']

# --- Perplexity補完 ---
def generate_response(context_question: str, context_answer: str, user_input: str) -> str:
    prompt = (
        f"以下はFAQに基づいたチャットボットの会話です。\n\n"
        f"質問: {context_question}\n回答: {context_answer}\n\n"
        f"ユーザーの質問: {user_input}\n\n"
        f"これを参考に、丁寧でわかりやすく自然な回答をしてください。"
    )

    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "sonar-pro",                  # 必要に応じて変更
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": prompt}
        ],
        "temperature": 1.0
    }

    try:
        res = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        res.raise_for_status()
        data = res.json()
        # Perplexityは OpenAI と違い、出力が data["output"][0]["content"][0]["text"] に格納される
        return data["output"][0]["content"][0]["text"]
    except requests.exceptions.RequestException as e:
        return f"APIリクエストエラー: {e}"
    except (KeyError, IndexError):
        return "Perplexity APIから予期しない形式のレスポンスが返ってきました。"

# --- チャットログ保存 ---
def save_log(log_data):
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"chatlog_{now}.csv"
    log_df = pd.DataFrame(log_data, columns=["ユーザーの質問", "チャットボットの回答"])
    log_df.to_csv(filename, index=False)
    return filename

# --- UI 初期設定 ---
st.title("LRADサポートチャット")
st.caption("※このチャットボットはFAQとAIをもとに応答しますが、すべての質問に正確に回答できるとは限りません。")

if 'chat_log' not in st.session_state:
    st.session_state.chat_log = []

with st.sidebar:
    st.markdown("### ⚙️ 表示設定")
    font_size = st.radio("文字サイズ", ["小", "標準", "大"], index=1)
    st.divider()
    st.markdown("背景色などの切り替え機能も追加できます")

font_size_map = {"小": "14px", "標準": "16px", "大": "20px"}
st.markdown(f"""
    <style>
    .chat-message {{
        font-size: {font_size_map[font_size]} !important;
    }}
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    .fixed-input {
        position: fixed;
        top: 30px;
        left: 0;
        width: 100%;
        background-color: #f9f9f9;
        padding: 10px;
        z-index: 999;
        border-bottom: 1px solid #ccc;
    }
    .chat-message {
        background-color: #e1f5fe;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .chat-message.assistant {
        background-color: #fff9c4;
    }
    </style>
""", unsafe_allow_html=True)

# ログ保存ボタン
if st.button("チャットログを保存"):
    filename = save_log(st.session_state.chat_log)
    st.success(f"チャットログを保存しました: {filename}")
    with open(filename, "rb") as f:
        st.download_button(
            label="このチャットログをダウンロード",
            data=f,
            file_name=filename,
            mime="text/csv"
        )

# 入力フォーム
st.markdown("<div class='fixed-input'>", unsafe_allow_html=True)
with st.form(key="chat_form", clear_on_submit=True):
    col1, col2 = st.columns([8, 1])
    with col1:
        user_input = st.text_input("質問をどうぞ：", key="user_input", label_visibility="collapsed")
    with col2:
        submitted = st.form_submit_button("送信")

    if submitted and user_input:
        if not is_valid_input(user_input):
            error_message = "エラーが発生しています。時間を空けてから再度お試しください。"
            st.session_state.chat_log.insert(0, (user_input, error_message))
            st.experimental_rerun()

        similar_q, similar_a = find_similar_question(user_input, faq_df)
        final_response = generate_response(similar_q, similar_a, user_input)
        st.session_state.chat_log.insert(0, (user_input, final_response))
        st.experimental_rerun()
st.markdown("</div>", unsafe_allow_html=True)

# チャット履歴表示
for user_msg, bot_msg in st.session_state.chat_log:
    with st.chat_message("user"):
        st.markdown(f"<div class='chat-message'>{user_msg}</div>", unsafe_allow_html=True)
    with st.chat_message("assistant"):
        st.markdown(f"<div class='chat-message assistant'>{bot_msg}</div>", unsafe_allow_html=True)
