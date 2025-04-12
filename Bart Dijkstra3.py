import json
import torch
import re
import time
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm  # ✅ 加入進度條

# ✅ 中文 BART 模型設定
print("🚀 正在載入 BART 模型...", flush=True)
model_name = "fnlp/bart-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
device = torch.device("cpu")
model = model.to(device)
print("✅ 模型載入完成！", flush=True)

# 讀取停用詞（每行一個詞）
stopwords_path = r"C:\Users\Leon\Desktop\Topic Model\本地環境\clearstopwords2.txt"
with open(stopwords_path, 'r', encoding='utf-8') as f:
    stopwords = set(line.strip() for line in f if line.strip())

# 讀取 LDA 主題的關鍵詞（lda_top_words.leiden.txt）
lda_keywords_path = r"C:\Users\Leon\Desktop\Topic Model\本地環境\lda_top_words.leiden.txt"
with open(lda_keywords_path, 'r', encoding='utf-8') as f:
    lda_keywords_raw = f.read().strip().split("\n")

# 分割成各個主題的詞彙（過濾1字詞）
lda_keywords = {}
for line in lda_keywords_raw:
    topic_id, words = line.split(":")
    words = words.strip()[1:-1].split(",")
    filtered_words = [word.strip().replace('"', '') for word in words if len(word.strip().replace('"', '')) >= 2]
    lda_keywords[int(topic_id.replace('Topic', '').strip())] = filtered_words

# 停用詞
combined_filter_words = stopwords

# 移除停用詞
def remove_filtered_phrases(text, filter_words):
    for word in filter_words:
        text = text.replace(word, "")
    return text

# 清洗文本
def clean_text_content(text, filter_words):
    text = remove_filtered_phrases(text, filter_words)
    text = re.sub(r"[^\u4e00-\u9fff，。！？]", "", text)
    return text

# 建立語意圖並找最短主題路徑 Dijkstra演算法
def get_dijkstra_topic_path(words):
    if len(words) < 2:
        return words
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(words)
    sim_matrix = cosine_similarity(tfidf_matrix)
    G = nx.Graph()
    for i, w1 in enumerate(words):
        for j, w2 in enumerate(words):
            if i != j:
                G.add_edge(w1, w2, weight=1 - sim_matrix[i, j])
    best_path = []
    min_cost = float("inf")
    for start in words:
        for end in words:
            if start != end and nx.has_path(G, start, end):
                try:
                    path = nx.dijkstra_path(G, start, end, weight='weight')
                    cost = nx.path_weight(G, path, weight='weight')
                    if cost < min_cost:
                        best_path = path
                        min_cost = cost
                except:
                    continue
    return best_path if best_path else words[:3]

# 加入 LDA 關鍵詞到文本開頭
def add_lda_keywords_to_text(text, lda_keywords, topic_id):
    lda_keywords_text = "、".join(lda_keywords.get(topic_id, []))
    return lda_keywords_text + "。" + text

# 讀取原始 JSON
file_path = r"C:\Users\Leon\Desktop\Topic Model\本地環境\topictexts_from_gamma.json"
with open(file_path, 'r', encoding='utf-8') as f:
    topic_texts = json.load(f)

# 判斷結構
entries = topic_texts.items() if isinstance(topic_texts, dict) else topic_texts
print(f"✅ 載入資料筆數：{len(entries)}", flush=True)

# 輸出路徑
output_path = r"C:\Users\Leon\Desktop\Topic Model\本地環境\bart dijkstra中文摘要結果.json"
summaries = []

print("🚀 開始處理資料...", flush=True)
start_time = time.time()  # ⏱️ 計時開始

# 逐筆處理文本（加上 tqdm 進度條）
for idx, item in tqdm(list(enumerate(entries)), desc="處理進度"):
    text = item[1] if isinstance(item, tuple) else item.get("text", "")
    if isinstance(text, list):
        text = " ".join(str(x) for x in text)
    elif not isinstance(text, str):
        continue

    topic_id = (idx % 30) + 1
    text = add_lda_keywords_to_text(text, lda_keywords, topic_id)
    text = clean_text_content(text, combined_filter_words)
    if not text.strip():
        continue

    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(device)
    summary_ids = model.generate(inputs["input_ids"], max_length=150, num_beams=6, temperature=1.0, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    topic_words = lda_keywords.get(topic_id, [])
    best_path = get_dijkstra_topic_path(topic_words)
    semantic_hint = "、".join(best_path)
    title_prompt = f"以下為一篇新聞摘要：{summary}。請根據內容幫這篇文章命名一個主題。"

    title_input = tokenizer(title_prompt, return_tensors="pt", max_length=128, truncation=True).to(device)
    title_ids = model.generate(title_input["input_ids"], max_length=30, num_beams=4, temperature=1.0, early_stopping=True) # max_length在這裡改參數 
    topic_title = tokenizer.decode(title_ids[0], skip_special_tokens=True)

    summaries.append({
        "id": topic_id,
        "summary": summary,
        "topic_title": topic_title
    })

end_time = time.time()  # ⏱️ 計時結束

# 儲存為 JSON
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(summaries, f, ensure_ascii=False, indent=2)

print("🎉 中文摘要與主題命名完成（含 Dijkstra 強化）！", flush=True)
print(f"⏱️ 總執行時間：{end_time - start_time:.2f} 秒", flush=True)
