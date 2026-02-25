# 我訓練 - 個人模型訓練實驗

個人進行的各種語言模型訓練實驗與測試。

## 📋 專案概述

本資料夾包含個人的模型訓練實驗，測試不同的預訓練策略、模型架構和資料集。

## 🤖 訓練模型

### 1. my-pretrained-2epochs/
- **訓練輪數**: 2 epochs
- **模型大小**: ~600MB
- **config.json**: 模型配置
- **generation_config.json**: 生成參數配置

### 2. my-pretrained-3epochs-YeungNLP-zh-ch/
- **訓練輪數**: 3 epochs
- **基礎模型**: YeungNLP (繁體中文優化)
- **壓縮包**: my-pretrained-3epochs-YeungNLP-zh-ch-20240108T101249Z-001.zip (689KB)

### 3. my-pretrained-5+5epochs-Langboat-zh-cn/
- **訓練輪數**: 5+5 epochs (兩階段訓練)
- **基礎模型**: Langboat BLOOM (簡體中文)

## 🎨 Gradio 應用

### gradio.py (6.2KB)
完整的對話應用，包含:
- 簡繁體自動轉換 (OpenCC)
- 串流輸出
- 對話歷史管理
- 可調整生成參數
- 預設範例問題

**主要功能**:
```python
- TextIteratorStreamer: 即時文字生成
- StoppingCriteria: 自訂停止條件
- 簡繁轉換: s2t / t2s
- 參數調整: temperature, top_p, top_k, repetition_penalty
```

### 10-10-gradio-app-對話模式streaming.py (6.3KB)
較新版本的 Gradio 應用，功能類似但有優化。

## 📊 訓練資料 (NLP/)

### YeungNLP 簡體中文資料集
`NLP/train_dataset_YeungNLP簡體/`
- **訓練集**: data_train/ (約 16MB)
- **驗證集**: data_val/ (約 241KB)
- **壓縮包**: train_dataset_YeungNLP簡體.zip (1.3MB)

### 訓練 Notebook
`NLP/w15-10-全微調-YeungNLP簡體.ipynb` (17KB)
- YeungNLP 模型全微調流程
- 簡體中文資料處理

## 📦 依賴套件

**requirements.txt** (1.5KB)
```
transformers
torch
gradio
opencc-python-reimplemented
sentencepiece
accelerate
```

## 🚀 快速開始

### 1. 安裝依賴
```bash
pip install -r requirements.txt
```

### 2. 啟動 Gradio 介面
```bash
python gradio.py
# 或
python 10-10-gradio-app-對話模式streaming.py
```

### 3. 訓練新模型
```bash
cd NLP/
jupyter notebook w15-10-全微調-YeungNLP簡體.ipynb
```

## ⚙️ Gradio 介面配置

### 預設範例問題
- 介紹哈利波特
- 各國首都問答
- 披薩製作教學
- 文章生成
- 新聞標題生成
- 機器學習演算法說明

### 可調參數
- **max_new_tokens**: 200 (生成長度)
- **temperature**: 1.0 (創造性)
- **top_p**: 0.95 (核心採樣)
- **top_k**: 200 (候選詞數量)
- **repetition_penalty**: 1.2 (重複懲罰)

## 🔧 模型比較

| 模型 | Epochs | 基礎模型 | 語言 | 大小 |
|------|--------|---------|------|------|
| my-pretrained-2epochs | 2 | Langboat | 簡體 | ~600MB |
| my-pretrained-3epochs-YeungNLP | 3 | YeungNLP | 繁體 | ~600MB |
| my-pretrained-5+5epochs-Langboat | 10 | Langboat | 簡體 | ~600MB |

## 📝 訓練筆記

### 觀察結果
1. **2 epochs**: 基本對話能力，有時會重複
2. **3 epochs**: 更流暢，較少重複
3. **5+5 epochs**: 最佳效果，回答更連貫

### 優化建議
- 增加 repetition_penalty 減少重複
- 調整 temperature 控制創造性
- 使用 top_p 和 top_k 平衡多樣性

## ⚠️ 注意事項

1. **模型路徑**: Gradio 腳本中的模型路徑需根據實際調整
2. **簡繁轉換**: 確保安裝 opencc-python-reimplemented
3. **GPU 記憶體**: 推理約需 2-4GB 顯存
4. **CUDA**: 自動偵測，無 GPU 會使用 CPU (較慢)

## 🎯 訓練目標

- [x] 基礎對話能力
- [x] 繁簡體中文支援
- [x] Web 介面展示
- [ ] 多輪對話記憶優化
- [ ] 特定領域知識增強

## 📂 測試記錄

`test.ipynb/`: 各種測試與實驗（空資料夾）

---
*訓練期間: 2023-12 ~ 2024-01*
*最後更新: 2024-01-08*
