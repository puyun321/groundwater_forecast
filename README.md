# grounwater_forecast
1. Daily-based data 為日資料，内有地下水、流量以及雨量資料
2. tenday-based data 為旬資料，内有地下水、流量以及雨量資料
3. data preprocessing 為資料前處理用途的程式碼，分別處理日資料以及旬資料
4. HBV_python 為HBV-light水文模型，用來模擬地下水的水位

''
HBV_python内，三個最重要的指標為：

a. Precipitation 降雨量
b. Temperature 溫度 (台灣可以假設成25°C，因爲台灣不會有融雪的情況)
c. ETpot 蒸發散
d. parameters 為水文模型會用到的參數，其中K值可向江老師要。這個值需要進行calibration
''
