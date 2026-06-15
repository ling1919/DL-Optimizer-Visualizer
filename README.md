## 專案簡介 (Project Description)
這是一個用於深度學習課程的開源專案。主要目的是實作並比較不同神經網路優化演算法（Optimizers）在相同資料集下的收斂速度與效能差異。

目前支援的優化演算法包含：
* **SGD (Stochastic Gradient Descent)** - 基礎梯度下降法
* **Momentum** - 加入動量機制的優化演算法
* **Adam** - 具備自適應學習率的進階優化演算法

## 檔案結構 (File Structure)
* `main.py`: 包含神經網路模型架構與三種優化器的核心實作程式碼。
* `data.csv`: 模型訓練用的二元分類資料集。

## 環境要求 (Requirements)
執行本程式需要安裝以下 Python 套件：
* `numpy`
* `pandas`

## 如何執行 (How to Run)
1. 確認資料夾內同時存在 `main.py` 與 `data.csv`。
2. 在終端機執行以下指令：
   python main.py
3. 程式將會自動輸出三種優化器在相同訓練次數下的最終 Loss 數值比較。
