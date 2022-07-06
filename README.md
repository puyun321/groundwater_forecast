# grounwater_forecast
```
此專案内各個檔案之敘述
1. Daily-based data 為日資料，内有地下水、流量以及雨量資料
2. tenday-based data 為旬資料，内有地下水、流量以及雨量資料
3. data preprocessing 為資料前處理用途的程式碼，分別處理日資料以及旬資料
4. HBV_python(raw_code) 為學長提供的HBV-light水文模型python代碼，用來模擬地下水的水位
5. HBV_python 為我更改HBV_python(raw_code)後可以為大家提供的程式輸入界面
6. lstm_HBV 為普運 研究方向(意要結合LSTM與HBV模型)
```

```
目前的工作目標：
a. 根據HBV_python 寫一個簡單的界面讓專題生也能夠使用
b. 修正一些參數 (如:台灣並不會融雪，所以此參數的影響應該為0)

未來的目標：
a. 由於目前的參數都是隨機生成的，所以需要根據觀測的地下水資料來修正參數
b. 利用ANN或其它演算法修正參數
```

# HBV-light model原理：<br>
![image](https://user-images.githubusercontent.com/41781189/177196537-f791cd21-7ab5-4977-bce7-c62fc27dce58.png)<br>
其它更詳細的内容請閲讀以下網址：<br>
http://www.gloh2o.org/hbv/?fbclid=IwAR3z6-TZ1_tiW9NLpTdB8dnKp1bufLjvBk6mnEr6JtqyihRD5k2sAJu7aJU <br>

# HBV-light model有三個最重要的指標以及其它參數<br>
a. Precipitation 降雨量<br>
b. Temperature 溫度 (台灣可以假設成25°C，因爲台灣不會有融雪的情況)<br>
c. ETpot 蒸發散<br>
d. parameters 為水文模型的參數<br>

其中，parameters的詳細内容：<br>
![image](https://user-images.githubusercontent.com/41781189/177193162-ad0a0090-cf7e-4e3f-93e2-bde627c53c03.png) <br>
其它更詳細的内容請閲讀以下網址：<br>
https://geomodeling.njnu.edu.cn/modelItem/5ad110bc-5655-438d-908d-ca8c6452c95f?fbclid=IwAR1MHzyImGV7kG4swr2iXeiREx6ILHbu6T5XffnrhL1TYCKhVcNFthI25V0
