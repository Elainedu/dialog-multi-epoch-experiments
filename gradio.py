"""
===============================================================================
專案名稱: Multi-turn Dialogue System - 多輪對話系統
===============================================================================

[專案簡介]
這是一個支援多輪對話的中文聊天機器人系統。系統可以記憶先前的對話內容，
根據完整的對話歷史生成連貫的回應。使用 Gradio Chatbot 介面提供友好的
對話體驗，並整合 OpenCC 進行繁簡轉換。

[核心技術]
- 模型: BLOOM-389M 預訓練模型（2 epochs 微調）
- 任務類型: Multi-turn Dialogue Generation（多輪對話生成）
- 深度學習框架: PyTorch + Transformers
- Web 介面: Gradio Chatbot
- 繁簡轉換: OpenCC

[多輪對話原理]
與單輪對話的差異:
- 單輪: 每次回應只基於當前問題，無記憶
- 多輪: 維護對話歷史，理解上下文關聯

實作方式:
1. 儲存所有歷史對話 [(user1, bot1), (user2, bot2), ...]
2. 每次生成時，將完整歷史拼接成 prompt
3. 格式: "Human: Q1\nAssistant: A1\nHuman: Q2\nAssistant: A2\nHuman: Q3\nAssistant:"
4. 模型基於完整上下文生成新回應

[技術特色]
1. 上下文記憶: 記住之前說過的話，產生連貫對話
2. 繁簡轉換: 自動處理簡體/繁體中文
3. 停止條件: 自定義 stopping criteria 控制生成結束
4. GPU 加速: 自動偵測並使用 GPU（如可用）
5. Gradio Chatbot: 提供類似聊天應用的介面

[啟動方式]
基本啟動:
    python gradio.py

[使用說明]
1. 啟動後開啟 http://127.0.0.1:7860
2. 在輸入框輸入訊息
3. 系統會基於對話歷史生成回應
4. 繼續對話，系統會記住之前的內容
5. 點擊「清除」可重置對話歷史

[對話範例]
User: 你好，請自我介紹
Bot: 我是一個AI助手，很高興為您服務...

User: 你剛才說你是什麼？（測試記憶）
Bot: 我剛才說我是一個AI助手...（能記住前一句）

User: 台灣有哪些景點？
Bot: 台灣有很多景點，例如...

User: 第一個景點怎麼去？（測試指代理解）
Bot: 要去台北101的話...（理解"第一個"指前面提到的景點）

[模型配置]
- 模型路徑: ./my-pretrained-2epochs
- 訓練: 基於 BLOOM-389M 微調 2 epochs
- 精度: float16 (GPU) / float32 (CPU)
- 最大生成長度: 根據對話歷史動態調整

[OpenCC 繁簡轉換]
系統整合 OpenCC 函式庫:
- s2t: Simplified to Traditional（簡轉繁）
- t2s: Traditional to Simplified（繁轉簡）

用途:
- 統一處理不同來源的中文文字
- 確保模型輸入格式一致

[Stopping Criteria]
自定義停止條件，當生成到以下情況時結束:
- 遇到 EOS (End of Sequence) token
- 生成特定停止符號
- 達到最大長度限制

[面試展示重點]
1. **多輪對話機制**: 解釋如何維護和使用對話歷史
2. **上下文理解**: 展示系統能理解代詞指代和話題延續
3. **記憶管理**: 討論長對話的記憶截斷策略
4. **實際應用**:
   - 客服機器人（需要記住客戶問題）
   - 虛擬助手（連續任務指令）
   - 教學輔助（根據先前回答調整解釋）
5. **技術挑戰**:
   - 對話歷史過長導致計算負擔
   - 如何選擇性保留重要歷史
   - 多輪對話的一致性維持

[檔案結構]
gradio.py                        # 本檔案 - 標準多輪對話介面
10-10-gradio-chat-streaming.py  # 串流版本（即時輸出）
my-pretrained-2epochs/           # 微調後的模型
    ├── config.json
    ├── pytorch_model.bin
    └── tokenizer files

[與串流版本的差異]
gradio.py (標準版):
- 等待完整生成後才顯示回應
- 適合短回應

10-10-gradio-chat-streaming.py (串流版):
- 邊生成邊顯示（打字機效果）
- 適合長回應，使用者體驗更好

[訓練資訊]
基礎模型: Langboat/bloom-389m-zh
微調資料: 中文對話資料集
訓練輪數: 2 epochs
微調方法: 全量微調 (Full Fine-tuning)

[對話歷史管理建議]
實際應用時的優化策略:
1. 滑動窗口: 只保留最近 N 輪對話
2. 摘要壓縮: 將早期對話摘要化
3. 關鍵資訊提取: 只保留重要的實體和事實
4. Token 限制: 控制總 token 數在模型限制內

[開發者]
碩士班課程專案 - 深度學習（進階）
建立日期: 2024

===============================================================================
"""

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
from threading import Thread
from transformers import GenerationConfig

from opencc import OpenCC
s2t = OpenCC('s2t')  # convert from Simplified Chinese to Traditional Chinese
t2s = OpenCC('t2s')  # convert from  Traditional Chinese to Simplified Chinese 

# https://www.gradio.app/guides/creating-a-chatbot-fast

CUDA_AVAILABLE = torch.cuda.is_available()
device = torch.device("cuda" if CUDA_AVAILABLE else "cpu")

model_name_or_path = './my-pretrained-2epochs'


if CUDA_AVAILABLE:
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype='auto')
else:
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

tokenizer = AutoTokenizer.from_pretrained(
    'Langboat/bloom-389m-zh',
    trust_remote_code=True,
    # llama不支持fast
    use_fast=True
)


class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [tokenizer.eos_token_id]
        #stop_ids = [29, 0]
        # if reply contains 'EOS' then we have reached the end of the conversation
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

def predict(message, history):

    history_transformer_format = history + [[message, ""]]
    stop = StopOnTokens()

    messages = "".join(["".join(["<Human>: \n"+item[0], "\n\n<Assistant>: \n"+item[1]]) for item in history_transformer_format])
    
    #轉成簡體字
    messages = t2s.convert(messages)
    #print(messages)
    
    # 可以採取最多只取4組對話，參考程式碼如下:
    # messages = ''.join([
    #     f"Human: {h[0]}\n\nAssistant: {'' if h[1] is None else h[1]}\n\n" for h in history[-4:]
    # ]).strip()
    
    model_inputs = tokenizer([messages], return_tensors="pt", add_special_tokens=False).to(device)
    # model_inputs = tokenizer([messages], return_tensors="pt").to(device)
    
    streamer = TextIteratorStreamer(tokenizer, timeout=10., skip_prompt=True, skip_special_tokens=True)
    
    
    generate_kwargs = dict(
        model_inputs,
        streamer=streamer,
        max_new_tokens= 200, #1024
        do_sample=True,
        top_p=0.95,
        top_k=200, #
        temperature=1.0,
        repetition_penalty=1.2, #
        num_beams=1, #目前只支援 =1
        stopping_criteria=StoppingCriteriaList([stop])
        )

    
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    # 輸出文字-----------------
    partial_message  = ""
    for new_token in streamer:
        
        partial_message += s2t.convert(new_token)
        # print(new_token)
        yield partial_message
        
        # 可在此額外設定跳開字元 似乎有許多<>這樣的符號會在文字中，可跳該此符號
        # if new_token != '<':
        #     partial_message += new_token
        #     yield s2t.convert(partial_message)

    #-----------



examples = [
    ["介紹一下哈利波特是什麼？"],
    ["法國的首都是什麼?"],
    ["英國的首都是什麼?"],
    ["中國的首都是什麼?"],
    ["你能不能詳細介紹一下怎麼做披薩？"],
    ["'下雨天'，請生成一篇文章。"],
    ["請生成一篇關於'下雨天'的文章。"],
    ["請生成一個新聞標題，描述一場正在發生的大型自然災害。"],
    ["為指定的詞彙創建一個關於該詞彙的簡短解釋。\n“人工智慧”"],
    ["編寫一篇簡短的新聞稿。\n新聞標題：一隻熊闖入市中心並在一棵樹上小憩"],
    ["列出3個不同的機器學習演算法，並說明它們的適用範圍。"],
    ["你是一個資深導遊，你能介紹一下中國的首都嗎?"],
    ["生成一篇關於人工智慧的200字文章，簡單介紹人工智慧的起源、應用和發展前景。\n"],
    ["針對給定的文本，生成一個摘要。摘要長度應該在100到200個字元之間。\n以下是一篇新聞報導的全文：\n北京時間7月23日消息，穀歌母公司Alphabet於當地時間週四發佈了該公司第二季度財報，超過了華爾街分析師的預期，一份財報顯示，Alphabet的二季度收入達到了382.1億美元，三個月淨利潤為72億美元，大幅超過市場預期，這主要歸功於線上廣告業務的快速增長。"],
    ]


# gr.ChatInterface(predict).queue().launch()
title = "自己微調的小型ChatGPT"
description = "微調GPT，讓它可以回答各式各樣的問題，就像人類對話一般"

            
chatinterface = gr.ChatInterface(fn=predict,
                                    examples=examples,
                                    title=title,
                                    description=description,
                                    # cache_examples=True, #事先算好example的答案
                                    textbox=gr.Textbox(value="請告訴我深度學習是甚麼?", placeholder="Ask me a question", container=False, lines=1, scale=5),
                                    
                                    #submit_btn=gr.Button("確定", scale=3),
                                    #stop_btn=gr.Button("中斷", scale=3),
                                    theme="soft",
                                    retry_btn="再產生一次答案",
                                    undo_btn="刪除最後一次對談",
                                    clear_btn="新的交談")               
chatinterface.queue()
chatinterface.launch()

# https://www.gradio.app/docs/chatinterface

    
'''
# 可修改成使用params以避免UserWarning訊息，尚未完成
generate_params = {
    "input_ids": input_ids,
    "max_new_tokens": max_new_tokens,
    "do_sample": do_sample,
    "temperature": temperature,
    "top_p": top_p,
    "top_k": top_k,
    "repetition_penalty": repetition_penalty,
    "typical_p": typical_p,
    "num_beams": num_beams,
    "stopping_criteria": transformers.StoppingCriteriaList(),
    "pad_token_id": tokenizer.pad_token_id,
}
'''