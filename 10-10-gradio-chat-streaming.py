"""
===============================================================================
專案名稱: Multi-turn Dialogue System with Streaming - 串流式多輪對話系統
===============================================================================

[專案簡介]
這是 gradio.py 的進階版本，增加了串流輸出功能。採用 TextIteratorStreamer
實現打字機效果，讓使用者在模型生成的同時就能看到輸出文字逐字出現，
大幅提升長回應時的使用者體驗。

[核心技術]
- 模型: BLOOM-389M 預訓練模型（2 epochs 微調）
- 任務類型: Multi-turn Dialogue with Streaming Output
- 串流技術: TextIteratorStreamer + Threading
- 深度學習框架: PyTorch + Transformers
- Web 介面: Gradio Chatbot
- 繁簡轉換: OpenCC

[串流輸出原理]
傳統方式 (gradio.py):
1. 模型完整生成所有 tokens
2. 全部生成完畢後才顯示
3. 長回應時使用者需等待

串流方式 (本檔案):
1. 使用 TextIteratorStreamer 建立生成迭代器
2. 在獨立 Thread 中執行模型生成
3. 主 Thread 從 streamer 逐字讀取
4. 每產生一個 token 就立即顯示
5. 打字機效果，即時反饋

[技術實作]
```python
# 建立 streamer
streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)

# 在獨立執行緒生成
generation_kwargs = {**inputs, "streamer": streamer, ...}
thread = Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()

# 主執行緒讀取並顯示
for new_text in streamer:
    accumulated_text += new_text
    yield accumulated_text  # 即時更新介面
```

[優勢對比]
指標          | 標準版       | 串流版
-------------|-------------|------------
首字延遲      | 高（需等全部完成）| 低（立即開始）
使用者體驗    | 等待焦慮     | 流暢自然
長回應適用性  | 差          | 優
實作複雜度    | 簡單         | 中等
CPU 使用     | 集中在生成時  | 分散在生成期間

[啟動方式]
啟動串流版本:
    python 10-10-gradio-chat-streaming.py

[使用說明]
1. 啟動後開啟 http://127.0.0.1:7860
2. 輸入訊息後立即看到回應逐字出現
3. 打字機效果讓等待過程更自然
4. 特別適合請求長回應（例如寫文章、解釋概念）

[適用場景]
串流版本特別適合:
1. 長文本生成 - 文章、故事、長篇說明
2. 程式碼生成 - 讓使用者看到生成過程
3. 翻譯任務 - 即時看到翻譯進度
4. 摘要生成 - 長文本的摘要輸出

標準版本適合:
1. 短回應 - 簡答、確認訊息
2. 精確生成 - 不希望使用者看到中間過程
3. 批次處理 - 不需即時顯示

[Threading 說明]
為何需要多執行緒:
- 模型生成是阻塞操作（blocking）
- 如在主執行緒生成，介面會凍結
- 獨立執行緒生成，主執行緒可即時更新介面

架構:
- Thread 1 (主執行緒): Gradio 介面 + 讀取 streamer
- Thread 2 (生成執行緒): 模型生成文字

[TextIteratorStreamer 原理]
Transformers 函式庫提供的工具類別:
- 作為生成器（generator），產出生成的 tokens
- 內部使用 queue 在執行緒間傳遞資料
- 自動處理 tokenization 和特殊 token
- skip_special_tokens=True 自動過濾 <eos>, <pad> 等

[面試展示重點]
1. **串流技術**: 說明如何實作即時文字生成
2. **Threading 應用**: 解釋多執行緒在 AI 應用中的使用
3. **使用者體驗**: 對比串流與非串流的 UX 差異
4. **技術權衡**:
   - 串流增加複雜度但改善體驗
   - 需注意執行緒安全問題
   - streamer 的 buffer 管理
5. **實際應用**: 討論 ChatGPT 等產品的串流實作

[演示技巧]
面試時的展示流程:
1. 先展示標準版 (gradio.py) - 長回應需等待
2. 再展示串流版 (本檔案) - 即時輸出
3. 對比使用者體驗差異
4. 說明技術實作細節

[程式碼關鍵部分]
生成函數的串流實作:
```python
def generate_response(message, history):
    # 建構對話歷史
    conversation = build_conversation(history, message)

    # 建立 streamer
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)

    # 準備生成參數
    inputs = tokenizer(conversation, return_tensors="pt").to(device)
    generation_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": 512,
        "temperature": 0.8,
    }

    # 在獨立執行緒生成
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # 逐字讀取並顯示
    partial_text = ""
    for new_text in streamer:
        partial_text += new_text
        yield partial_text  # Gradio 會自動更新介面

    thread.join()  # 等待生成完成
```

[注意事項]
1. streamer 需在主執行緒建立，在生成執行緒使用
2. 必須使用 yield 而非 return 才能實現串流
3. thread.join() 確保生成完全結束
4. 錯誤處理要同時考慮兩個執行緒

[效能考量]
- 串流不會影響生成速度
- 略微增加記憶體使用（buffer）
- 網路傳輸更均勻（非一次性大量資料）

[檔案結構]
10-10-gradio-chat-streaming.py   # 本檔案 - 串流版多輪對話
gradio.py                        # 標準版多輪對話
my-pretrained-2epochs/           # 微調後的模型

[改進方向]
1. 加入生成取消功能（stop generation）
2. 顯示生成速度（tokens/sec）
3. 支援多種串流模式（字元級、詞級、句子級）
4. 加入生成進度條

[開發者]
碩士班課程專案 - 深度學習（進階）
建立日期: 2024
特色: 串流輸出優化使用者體驗

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

#model_name_or_path = 'my-pretrained-2epochs-langboat'
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

    messages = "".join(["".join(["<Human>: \n"+item[0], "\n\n<Assistant>:"+item[1]]) for item in history_transformer_format])
    
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
        max_new_tokens= 400, #1024
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