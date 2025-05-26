import streamlit as st
import pandas as pd
import requests
from rapidfuzz import fuzz, process

# シークレットから APIキーを取得
API_KEY = st.secrets["PerplexityAPI"]["api_key"]
API_URL = "https://api.perplexity.ai/chat/completions"
MODEL = "llama-3-sonar-small-32k-online"

# FAQ の読み込み
faq_df = pd.read_csv("faq.csv")

def find_similar_question(user_input, faq_df):
    questions = faq_df["質問"].tolist()
    best_match = process.extractOne(user_input, questions, scorer=fuzz.token_sort_ratio)
    if best_match[1] >= 40:
        answer = faq_df.loc[faq_df["質問"] == best_match[0], "回答"].values[0]
        return best_match[0], answer
    return None, None

def generate_answer(user_input, reference_info):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    messages = [
        {"role": "system", "content": "あなたは熱分解装置LRADの専門家です。"},
        {"role": "user", "content": f"以下の参考情報をもとに回答してください。\n\n{reference_info}\n\n質問: {user_input}"}
    ]

    payload = {
        "model": MODEL,
        "messages": messages
    }

    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code != 200:
        st.error(f"Perplexity API エラー: {response.status_code} {response.reason}")
        return None

    return response.json()["choices"][0]["message"]["content"]

# Streamlit UI
st.title("LRADサポートチャット（Perplexity API）")
user_input = st.text_input("質問を入力してください:")

if st.button("送信") and user_input:
    similar_q, reference_info = find_similar_question(user_input, faq_df)
    if reference_info:
        st.markdown(f"**参考情報（FAQより）:**\n> {reference_info}")
    else:
        reference_info = "特に参考情報は見つかりませんでした。"
        st.markdown("（参考情報は見つかりませんでした）")

    answer = generate_answer(user_input, reference_info)
    if answer:
        st.markdown("### 回答:")
        st.write(answer)
