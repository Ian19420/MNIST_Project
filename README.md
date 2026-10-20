手寫數字辨識系統說明文件

一、 系統架構(System Architecture)
本專案採用B/S(Browser/Server)架構，整合深度學習模型與網頁前端介面，實現即時手寫數字辨識。

1. 開發環境與技術棧
程式語言：Python 3.10
核心框架：TensorFlow/Keras(深度學習模型訓練與推論)
前端介面：Streamlit(Web GUI開發)
影像處理：OpenCV(cv2), PIL, NumPy(影像前處理)
繪圖工具：Streamlit Drawable Canvas (手寫板整合)

2. 檔案目錄結構
│
├── models/
│   ├── mnist_model.keras      #訓練完成的CNN模型權重檔
│   └── MNIST_Training.ipynb   #模型訓練腳本
│
├── src/
│   └── app.py                 #系統主程式(Streamlit介面與邏輯)
│
└── README.md                  #系統說明文件(本檔)

3. AI模型設計(CNN)
本系統捨棄傳統全連接層(Dense/MLP)，改採用卷積神經網路(CNN)，以提升對手寫筆跡的平移與變形容忍度。

Input Layer：(28, 28, 1)灰階影像。
Feature Extraction(特徵提取)：
  Conv2D：使用3x3卷積核提取邊緣與幾何特徵。
  MaxPooling2D：進行降維，保留關鍵特徵並減少運算量。
Classification(分類器)：
  Flatten：將特徵圖攤平。
  Dense(Softmax)：輸出0~9個類別的機率分佈。

二、執行方式(Execution Method)
若需在其他電腦上執行本系統，請確保以下完成步驟：
步驟1：安裝必要套件
  請確保Python環境已安裝，並確保有以下依賴庫：
    tensorflow streamlit streamlit-drawable-canvas opencv-python pandas numpy pillow
步驟2：啟動系統開啟終端機(Terminal/CMD)，切換至專案根目錄，並執行以下指令：
  streamlit run src/app.py
步驟3：操作介面系統啟動後，瀏覽器將自動開啟。
  在左側黑色畫布區使用滑鼠書寫數字 0~9。點擊「開始辨識」按鈕。右側將即時顯示預測數字、信心水準及機率分佈圖。

三、執行流程(Execution Flow)
本系統的核心在於「模擬MNIST資料集製程」的影像前處理技術，確保使用者手寫的圖片能轉換成模型看得懂的格式。
  階段1：前端輸入(Input)
    使用者透過st_canvas在350x350像素的畫布上繪製數字。
    系統即時獲取畫布的影像數據(Image Data)。
  階段2：影像前處理(Preprocessing)-核心關鍵
    為解決手寫位置偏移及筆觸粗細問題，系統執行以下演算法：
      灰階轉換：將RGB影像轉換為灰階(Grayscale)。
      色彩反轉：若偵測到白底黑字，自動反轉為黑底白字(符合MNIST格式)。
      邊框偵測(Bounding Box)：透過np.where找出筆跡像素的上下左右邊界。
      自動裁切(Crop)：切除多餘黑邊，只保留數字本體。
      等比例縮放(Resize)：將裁切後的影像縮放至長邊為20像素，保持長寬比，避免變形。
      置中補邊(Padding)：將縮放後的影像貼至28x28畫布的正中央，周圍補黑邊。
      二值化增強(Thresholding)：使用OTSU或固定閥值將模糊邊緣銳利化，消除灰色雜訊。
  階段3：AI推論(Inference)
    將處理好的28x28影像Reshape為(1, 28, 28, 1)。
    正規化像素值至0~1之間(Float32)。
    輸入CNN模型進行預測，取得Logits輸出。
  階段4：結果展示(Output)
    透過np.argmax取出機率最高的數字。
    使用Streamlit的st.metric與st.bar_chart視覺化呈現信心度與各數字機率。

四、附註
  AI輔助聲明：本專案之UI介面優化與OpenCV前處理邏輯，開發過程中使用大型語言模型(Gemini)輔助除錯與架構建議。