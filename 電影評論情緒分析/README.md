## 目標 : 
透過深度學習模型，分析電影評論情感，以理解真實評論傾向，協助建立更精確的評論系統。

## 數據 :
來源：Aidea（人工智慧共創平台）所提供。
訓練資料：共23472筆，取其中10% 作為驗證資料。
           其中 正面：11963筆 (51%)、負面：11509筆 (49%)。
測試資料：共5869筆。

## 模型 :
1. Bidirectional LSTM
2. word2vec + BiLSTM

## 方法 :
1. 文字前清理：
   移除html標記、標點符號、停用詞等，並進行詞幹還原。<br>
   將處理後的文本轉換成序列向量，再進行截斷或填充，以確保相同長度的輸入數據。
2. 模型與訓練：
   使用LSTM模型進行情感分類預測。  
   採用Dropout和Early stopping技術，以減少過度擬合的風險。
4. 性能改進：
   調整LSTM模型為雙向結構，使其能夠更好地理解文本的上下文。
   使用預先訓練的word2vec詞嵌入，進而提升模型的性能。


## 結果 :
1. LSTM	:                F1 score 為 0.7927，	Accuracy 為 0.7930， AUC 為 0.7933
2. LSTM + Early stopping :  F1 score 為 0.8381，	Accuracy 為 0.8378， AUC 為 0.8381
3. Bidirectional  LSTM :    F1 score 為 0.8429，	Accuracy 為 0.8424， AUC 為 0.8426
4. word2vec + BiLSTM	:     F1 score 為 0.8759，	Accuracy 為 0.8727， AUC 為 0.8726


