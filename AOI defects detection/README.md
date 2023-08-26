# AOI_defects_detection

## 專案介紹
此專案為藉由AOI影像訓練深度學習模型辨識產品表面瑕疵，將其進行分類。

## 資料來源
AOI影像資料由工研院電光所在Aidea(人工智慧共創平台)釋出作為開放性議題，供參賽者建立瑕疵辨識模型。
資料來源：https://aidea-web.tw/topic/285ef3be-44eb-43dd-85cc-f0388bf85ea4

## 基本資訊
- 訓練資料：共2528張，並取其中20%作為驗證資料
- 測試資料：共10142張
- 影像類別：共6類(正常類別 + 5種瑕疵類別)
- 影像尺寸：512x512

## 資料處理
- 影像水平、垂直平移0.05
- 影像水平、垂直翻轉
- 影像大小縮為 224 x 224

## 模型
- Xception
- DenseNet169

## 模型結果
Model	        Training	  Validation	  Testing

Xception      	99.70%	    99.41%	    99.36%

DenseNet169	    100%	      99.60%	    99.53%
