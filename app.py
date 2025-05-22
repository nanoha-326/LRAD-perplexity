"""LRADサポートチャット  
Perplexity Labs API 版（OpenAI 不使用）
────────────────────────────────────────
必要パッケージ:
  pip install streamlit pandas numpy scikit-learn sentence-transformers requests

※APIキーは .streamlit/secrets.toml に以下のように保存してください。

[PerplexityAPI]
api_key = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
"""

# ── Imports ────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import re
import unicodedata
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ── Streamlit 設定 ────────────────────────────────────────────────────────
st.set_page_config(page_title="LRADサポートチャット", page_icon="📘", layout="centered")

# ── Embedding モデル初期化（ローカル: MiniLM）────────────────────────────
@st.cache_resource(show_spinner=False)
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# ── Perplexity API 共通設定 ─────────────────────────────────────────────
PPLX_ENDPOINT = "https://api.perplexity.ai/chat/completions"
PPLX_HEADERS = {
    "Authorization": f"Bearer {st.secrets['PerplexityAPI']['api_key']}",
    "Content-Type": "application/json",
}

# ── システムプロンプト ───────────────────────────────────────────────────
system_prompt = """
あなたはLRAD専用のチャットボットです。
「LRAD（エルラド）」とは熱分解装置（遠赤外線電子熱分解装置）のことで、これは有機廃棄物の処理装置です。
あなたの役割は、この装置の検証をサポートすることです。

以下の点を守ってください：
・装置に関連することのみを答えてください。
・関係ない話題（天気、芸能、スポーツなど）には答えないでください。
・FAQにない場合は「わかりません」と丁寧に答えてください。
"""

# ── 入力バリデーション ────────────────────────────────────────────────
def is_valid_input(text: str) -> bool:
    text = text.strip()
    if not (3 <= len(text) <= 300):
        return False
    non_alpha_ratio = len(re.findall(r"[^A-Za-z0-9ぁ-んァ-ヶ一-龠\s]", text)) / len(text)
    if non_alpha_ratio > 0.3:
        return False
    try:
        unicodedata.normalize("NFKC", text).encode("utf-8")
    except UnicodeError:
        return False
    return True

# ── 埋め込み生成 ────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_embedding(text: str) -> np.ndarray:
    """returns L2‑normalised vector"""
    emb = embedder.encode(text, normalize_embeddings=True)
    return emb

# ── FAQ（埋め込み付き）ロード ────────────────────────────────────────────
@st.cache_data
def load_faq(csv_path: str):
    df = pd.read_csv(csv_path)
    df["embedding"] = df["質問"].apply(get_embedding)
    return df

faq_df = load_faq("faq.csv")

# ── 類似質問検索 ────────────────────────────────────────────────────────
def find_similar_question(user_input: str, faq_df: pd.DataFrame):
    user_vec = get_embedding(user_input)
    faq_vecs = np.stack(faq_df["embedding"].values)
    scores = cosine_similarity([user_vec], faq_vecs)[0]
    idx = scores.argmax()
    return faq_df.iloc[idx]["質問"], faq_df.iloc[idx]["回答"]

# ── Perplexity API 呼び出し ─────────────────────────────────────────────
def perplexity_chat(messages: list[dict], model: str = "sonar-small-chat", temperature: float = 1.2) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    resp = requests.post(PPLX_ENDPOINT, headers=PPLX_HEADERS, json=payload, timeout=60)
    if resp.status_code != 200:
        return f"エラー: Perplexity API が {resp.status_code} を返しました"
    data = resp.json()
    # 互換性取りのため choices / output 両方を考慮
    if "choices" in data:
        return data["choices"][0]["message"]["content"].strip()
    if "output" in data:
        try:
            return data["output"][0]["content"][0]["text"].strip()
        except (KeyError, IndexError):
            pass
    return "エラー: 予期しないレスポンス形式です。"

# ── GPT応答生成（Perplexity）───────────────────────────────────────────
def generate_response(context_q: str, context_a: str, user_input: str) -> str:
    prompt = (
        "以下はFAQに基づいたチャットボットの会話です。\n\n"
        f"質問: {context_q}\n回答: {context_a}\n\n"
        f"ユーザーの質問: {user_input}\n\n"
        "これを参考に、丁寧でわかりやすく自然な回答をしてください。"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    return perplexity_chat(messages, model="sonar-small-chat", temperature=1.2)

# ── ログ保存 ────────────────────────────────────────────────────────────
def save_log(log_data):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"chatlog_{ts}.csv"
    pd.DataFrame(log_data, columns=["ユーザーの質問", "チャットボットの回答"]).to_csv(
        fname, index=False, encoding="utf-8-sig"
    )
    return fname

# ── UI ──────────────────────────────────────────────────────────────────
st.title("LRADサポートチャット")
st.caption("※このチャットボットはFAQとAIをもとに応答しますが、すべての質問に正確に回答できるとは限りません。")

if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

# サイドバーの設定
with st.sidebar:
    st.header("⚙️ 設定変更")
    size_option = st.radio("文字サイズを選択", ["小", "中", "大"], index=1)

size_map = {"小": 18, "中": 22, "大": 30}
font_px = size_map[size_option]

# 動的 CSS
st.markdown(
    f"""
    <style>
    div.stChatMessage p {{font-size: {font_px}px !important;}}
    h1 {{font-size: {font_px + 10}px !important;}}
    p > small {{font-size: {max(font_px - 4, 10)}px !important;}}
    div.stText, div[data-testid="stMarkdownContainer"] > div p {{font-size: {font_px}px !important;}}
    div.stTextInput > div > input::placeholder {{font-size: {font_px}px !important;}}
    div.stTextInput > div > input {{font-size: {font_px}px !important;}}
    </style>
    """,
    unsafe_allow_html=True,
)

# ログ保存ボタン
if st.button("チャットログを保存"):
    fname = save_log(st.session_state.chat_log)
    st.success(f"チャットログを保存しました: {fname}")
    with open(fname, "rb") as f:
        st.download_button("このチャットログをダウンロード", data=f, file_name=fname, mime="text/csv")

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
