import streamlit as st
import pandas as pd
import numpy as np
import datetime
import re
import unicodedata
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI  # 新バージョン用

# --- Streamlitの設定 ---
st.set_page_config(page_title="LRADサポートチャット", page_icon="📘", layout="centered")

# --- OpenAIクライアントの初期化 ---
client = OpenAI(api_key=st.secrets.OpenAIAPI.openai_api_key)

# --- システムプロンプト ---
system_prompt = """
あなたはLRAD専用のチャットボットです。
「LRAD（エルラド）」とは熱分解装置（遠赤外線電子熱分解装置）のことで、これは有機廃棄物の処理装置です。
あなたの役割は、この装置の検証をサポートすることです。

以下の点を守ってください：
・装置に関連することのみを答えてください。
・関係ない話題（天気、芸能、スポーツなど）には答えないでください。
・FAQにない場合は「わかりません」と丁寧に答えてください。
"""

# --- 入力バリデーション ---
def is_valid_input(text: str) -> bool:
    text = text.strip()
    if len(text) < 3 or len(text) > 300:
        return False
    non_alpha_ratio = len(re.findall(r'[^A-Za-z0-9ぁ-んァ-ヶ一-龠\s]', text)) / len(text)
    if non_alpha_ratio > 0.3:
        return False
    try:
        unicodedata.normalize('NFKC', text).encode('utf-8')
    except UnicodeError:
        return False
    return True

# --- 埋め込み生成 ---
def get_embedding(text):
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-ada-002"
    )
    return np.array(response.data[0].embedding)

# --- FAQロード（埋め込み付き） ---
@st.cache_data
def load_faq(csv_file):
    df = pd.read_csv(csv_file)
    df['embedding'] = df['質問'].apply(lambda x: get_embedding(x))
    return df

faq_df = load_faq("faq.csv")

# --- 類似質問検索 ---
def find_similar_question(user_input, faq_df):
    user_vec = get_embedding(user_input)
    faq_vecs = np.stack(faq_df['embedding'].values)
    scores = cosine_similarity([user_vec], faq_vecs)[0]
    top_idx = scores.argmax()
    return faq_df.iloc[top_idx]['質問'], faq_df.iloc[top_idx]['回答']

# --- GPT応答生成 ---
def generate_response(context_q, context_a, user_input):
    prompt = f"以下はFAQに基づいたチャットボットの会話です。\n\n質問: {context_q}\n回答: {context_a}\n\nユーザーの質問: {user_input}\n\nこれを参考に、丁寧でわかりやすく自然な回答をしてください。"
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=1.2
    )
    return response.choices[0].message.content

# --- ログ保存 ---
def save_log(log_data):
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"chatlog_{now}.csv"
    pd.DataFrame(log_data, columns=["ユーザーの質問", "チャットボットの回答"]).to_csv(filename, index=False, encoding='utf-8-sig')
    return filename


# --- UI ---
st.title("LRADサポートチャット")
st.caption("※このチャットボットはFAQとAIをもとに応答しますが、すべての質問に正確に回答できるとは限りません。")

if 'chat_log' not in st.session_state:
    st.session_state.chat_log = []

# --- サイドバーに設定項目を追加 ---
with st.sidebar:
    st.header("⚙️ 設定変更")
    size_option = st.radio(
        "文字サイズを選択",
        ["小", "中", "大"],
        index=1,  # デフォルトは「中」
        horizontal=False
    )

# --- 選択されたサイズに応じたCSSを反映 ---
size_map = {
    "小": 18,
    "中": 22,
    "大": 30
}
font_px = size_map[size_option]

st.markdown(
    f"""
    <style>
    /* チャットメッセージ */
    div.stChatMessage p {{
        font-size: {font_px}px !important;
    }}

    /* タイトル (st.title() の h1タグ) */
    h1 {{
        font-size: {font_px + 10}px !important;
    }}

    /* キャプション (p > small) */
    p > small {{
        font-size: {max(font_px - 4, 10)}px !important;
    }}

    /* 通常の本文テキスト */
    div.stText, div[data-testid="stMarkdownContainer"] > div p {{
        font-size: {font_px}px !important;
    }}
        /* テキスト入力のプレースホルダー文字サイズ */
    div.stTextInput > div > input::placeholder {{
        font-size: {font_px}px !important;
    }}

    /* テキスト入力の中の文字サイズ */
    div.stTextInput > div > input {{
        font-size: {font_px}px !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ログ保存ボタン
if st.button("チャットログを保存"):
    filename = save_log(st.session_state.chat_log)
    st.success(f"チャットログを保存しました: {filename}")
    with open(filename, "rb") as f:
        st.download_button("このチャットログをダウンロード", data=f, file_name=filename, mime="text/csv")

# 入力フォーム
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_area("質問をどうぞ：", height=100)
    submitted = st.form_submit_button("送信")

if submitted and user_input:
    if not is_valid_input(user_input):
        st.session_state.chat_log.insert(0, (user_input, "エラー：入力が不正です。"))
        st.experimental_rerun()

    with st.spinner("回答生成中…お待ちください。"):
        similar_q, similar_a = find_similar_question(user_input, faq_df)
        answer = generate_response(similar_q, similar_a, user_input)

    st.session_state.chat_log.insert(0, (user_input, answer))
    st.experimental_rerun()

# チャット履歴表示
for user_msg, bot_msg in st.session_state.chat_log:
    with st.chat_message("user"):
        st.markdown(user_msg)
    with st.chat_message("assistant"):
        st.markdown(bot_msg)
