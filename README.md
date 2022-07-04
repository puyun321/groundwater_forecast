# grounwater_forecast
1. Daily-based data 為日資料，内有地下水、流量以及雨量資料
2. tenday-based data 為旬資料，内有地下水、流量以及雨量資料
3. data preprocessing 為資料前處理用途的程式碼，分別處理日資料以及旬資料
4. HBV_python 為HBV-light水文模型，用來模擬地下水的水位

```
HBV_python内，三個最重要的指標為：

a. Precipitation 降雨量
b. Temperature 溫度 (台灣可以假設成25°C，因爲台灣不會有融雪的情況)
c. ETpot 蒸發散
d. parameters 為水文模型的參數
參數的詳細内容：
![image](https://user-images.githubusercontent.com/41781189/177193162-ad0a0090-cf7e-4e3f-93e2-bde627c53c03.png)
其它更詳細的内容請閲讀以下網址：
https://geomodeling.njnu.edu.cn/modelItem/5ad110bc-5655-438d-908d-ca8c6452c95f?fbclid=IwAR1MHzyImGV7kG4swr2iXeiREx6ILHbu6T5XffnrhL1TYCKhVcNFthI25V0
```
