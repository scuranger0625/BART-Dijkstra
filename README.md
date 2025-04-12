# 🧠 BART-Dijkstra: 語意引導式摘要與主題命名演算法

**BART-Dijkstra** 是由 [`scuranger0625`](https://github.com/scuranger0625) 於 2025 年提出的啟發式演算法，結合圖論中經典的 **Dijkstra 最短路徑演算法** 與自然語言生成模型 **BART**，實現針對主題語意圖的引導式摘要與命名任務。

---

## 📌 方法簡介

本演算法主要應用於主題建模後的語意處理流程，具體流程如下：

1. 使用 LDA 模型產生每個主題的關鍵詞（`topicwords`）。
2. 利用 TF-IDF 建構主題語意圖，節點為關鍵詞，邊權為語意相似度。
3. 透過 Dijkstra 演算法，在語意圖中尋找一條「語意最短路徑」作為提示詞序列。
4. 將此序列餵入 BART 模型，引導其生成摘要與主題命名。

此演算法同時保有：
- BART 的語言生成能力
- Dijkstra 的語意聚焦導引能力

---

## 🧪 應用場景

- 主題模型之後的語意摘要生成（例如新聞、事實查核、論壇文本）
- 自動化主題命名
- 結合網絡圖與生成式模型的語意處理應用

---

## 🛠️ 執行環境

- Python 3.10+
- PyTorch
- Transformers (`transformers==4.x`)
- NetworkX
- scikit-learn
- tqdm

安裝方式（建議使用虛擬環境）：
```bash
pip install -r requirements.txt

