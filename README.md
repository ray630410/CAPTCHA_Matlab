##  網頁驗證碼(CAPTCHA)辨識模型訓練 (CNN)
### 摘要
這個報告旨在開發一個基於卷積神經網絡（CNN）的模型，用於自動識別網頁驗證碼（CAPTCHA），進而實現自動登入特定網站並抓取數據。首先，我通過爬蟲技術從目標網站下載CAPTCHA圖片，以建立訓練數據集。接著，我使用MatLab建立和訓練的一個深度學習模型，該模型能夠從圖片中識別出驗證碼。我利用它對我的目標站上爬下來的圖片進行了訓練及圖片預測測試。然後將預測的程式整合到自動登入的腳本中。最終，我成功實現了自動識別驗證碼並登入網站的功能，這對於我想要自動化數據收集和處理具有很大的意義。我的報告提供了詳細的方法、實踐用的程式碼，以及關於CNN模型的一些學習心得。
### 動機
許多網站為了避免非人為操作的存取頻繁增加了網站的流量負擔，因此建立了驗證碼機制來限制網站只有人可以進行操作。但是身為一個想善用資訊網路替我們服務的現代人，我們常常對於一個自己擁有帳號密碼的網站，有自動前往讀取資料的需求。
例如：我們有數個銀行的存款帳戶跟信用卡，同時也有網路銀行的帳密，我們希望能自動將各銀行帳戶中的存款，信用卡消費總額同步到我們的個人資產負債表中方便一目瞭然的管理個人資產。
### 步驟
- 爬取並下載目標網站中的CAPTCHA圖片方便進行訓練(fetch.m)
- 人工標記圖片
- 撰寫訓練程式並將訓練好的模型儲存下來(train.m)
- 撰寫預測程式並載入訓練好的模型來對圖片進行辨識(test.m)
- 爬取並下載目標網站中的CAPTCHA圖片辨識後存檔(更新後的fetch.m)
- 檢查修正標記圖片重新訓練
- 整合成自動登入網站的程式(login.m)
### 簡單認識網頁驗證碼(CAPTCHA)
#### 目標網頁範例：
https://eze8.npa.gov.tw/NpaE8ServerRWD/CL_Query.jsp
#### 用動態生成CAPTCHA圖片的網址範例：
https://eze8.npa.gov.tw/NpaE8ServerRWD/CheckCharImgServlet
### 步驟實作
#### 爬取並下載目標網站中的CAPTCHA圖片方便進行訓練(fetch.m)

- 建立下載圖片儲存的資料夾
```matlab=
% Create a directory 'download' if it does not exist
folderPath = './download';
if ~exist(folderPath, 'dir')
    mkdir(folderPath);
end
```

- 呼叫下載圖片並儲存的函式100次，每次間隔3秒避免被當作攻擊
```matlab=
% Download and save 100 images
successCount = 0;
for k = 1:100  % Adjust the range if you want more or fewer images
    if downloadCaptcha(folderPath)
        successCount = successCount + 1;
    end
    pause(3);  % Wait for 3 seconds before downloading the next image
end

disp(['Successfully downloaded ' num2str(successCount) ' images.']);
```
- 建立下載圖片並儲存的函式
```matlab=
% Function to download and save one CAPTCHA image
function success = downloadCaptcha(folderPath)
    options = weboptions('ContentType', 'image', 'Timeout', 30);
    captchaUrl = '此處使用動態生成CAPTCHA圖片的網址';
    try
        imdata = webread(captchaUrl, options);
        randomCode = randi([0, 10000]);  % Generate a random integer between 0 and 10000
        filePath = fullfile(folderPath, sprintf('%d.jpg', randomCode));      
        %use predict instead of random
        imwrite(imdata, filePath);
        success = true;
    catch
        success = false;
end

#### 撰寫訓練程式並將訓練好的模型儲存下來(train.m)

- 定義一些訓練時要使用的參數
```matlab=
% Initialize Variables
epochs = 10;       % Number of epochs for training
digitsInImg = 4;   % CAPTCHA length
numClasses = 36;   % 26 English alphabets + 10 digits
xList = {};        % Store image data
yList = [];        % Store character
```
- 建立數字字元及大寫英文字元對應到數字的對應表
```matlab=
% Create map from characters to integers
characters = ['0':'9', 'A':'Z'];  % Digits + uppercase letters
charToInt = containers.Map('KeyType', 'char', 'ValueType', 'int32');
for i = 1:length(characters)
    charToInt(characters(i)) = i;
end
```
- 將training資料夾中的所有圖片分割影像資料放進x_list，從檔名取得字元對應的數字放進y_list
```matlab=
% Load and Process Images
imgFiles = dir('training/*.jpg');
for idx = 1:length(imgFiles)
    imgFilename = imgFiles(idx).name;
    img = imread(fullfile('training', imgFilename));
    img = rgb2gray(img);  % Assuming the images need to be grayscale
    imgArray = double(img);
    [imgRows, imgCols] = size(imgArray);
    [xList, yList] = splitDigitsInImg(imgArray, imgFilename, xList, yList, charToInt, digitsInImg);
end
```
- 將y_list轉換成二維陣列，本來每一個元素就變成一個由0,1組成的，36個元素的陣列
```matlab=
% Convert Labels to Categorical
yList = categorical(yList, 1:numClasses);
```

- 將資料分成訓練組跟測試組
```matlab=
% Split Data into Training and Testing Sets
perm = randperm(numel(xList));
idxTrain = perm(1:round(0.9*numel(xList)));
idxTest = perm(round(0.9*numel(xList))+1:end);
xTrain = xList(idxTrain);
yTrain = yList(idxTrain);
xTest = xList(idxTest);
yTest = yList(idxTest);
```

- 建立模型
```matlab=
% Define Network Architecture
layers = [
    imageInputLayer([imgRows, imgCols/digitsInImg, 1])
    convolution2dLayer(3, 32, 'Padding', 'same')
    reluLayer()
    convolution2dLayer(3, 64, 'Padding', 'same')
    reluLayer()
    maxPooling2dLayer(2, 'Stride', 2)
    dropoutLayer(0.25)
    fullyConnectedLayer(128)
    reluLayer()
    dropoutLayer(0.5)
    fullyConnectedLayer(numClasses)
    softmaxLayer()
    classificationLayer()
];

---
**二、對程式逐行進行說明：**
  - layers = [];
這行創建了一個Sequential模型的layers陣列。Sequential模型是一系列層的線性堆疊，每層恰有一個輸入張量和一個輸出張量，而每一層都定義在這個陣列中。
  - imageInputLayer([imgRows, imgCols/digitsInImg, 1])
定義了模型輸入數據的形狀。這裡假設每個輸入圖像的尺寸為`img_rows`行，`img_cols / digits_in_img`列，並且是單通道（灰度圖像）。
  - convolution2dLayer(3, 32, 'Padding', 'same')
這行向模型中添加了一個卷積層，先使用32個過濾器的用意是先做比較粗糙的抽象化，也就是說先將比較大的特徵值進行強化。各參數意義如下：
    - `3`定義了每個過濾器的大小，即3x3。
    - `32`表示該層有32個過濾器（或稱卷積核）。
    - `Padding` 設為 `same` 時，意味著 MATLAB 會自動添加足夠的填充（padding），使得輸出特徵圖的尺寸等於輸入特徵圖的尺寸，假設步長（stride）為 1。如果步長大於 1，輸出尺寸會相應減小，但填充依然保證在每個方向上是對稱的，以減少邊界效應。。
  - reluLayer()
添加了一個使用 Rectified Linear Unit (ReLU) 激活函数的層。
  - convolution2dLayer(3, 64, 'Padding', 'same')
這行添加了第二個卷積層，跟第一層唯一不同的是有64個過濾器，將特徵做比較細緻的抽象化。
卷積神經網路中，淺層通常使用較少的過濾器，用來將較大的特徵先訓練出來，接下來深層就會使用較多的過濾器，以便訓練模型能識別更細微的特徵。
  - reluLayer()
添加了一個使用 Rectified Linear Unit (ReLU) 激活函数的層。
  - maxPooling2dLayer(2, 'Stride', 2)
這行添加了一個最大池化層，用於降低特徵圖的空間尺寸，減少參數和計算量，同時也有助於防止過擬合。第一個pool_size是2，表示池化窗口的大小為2x2。接下來'Stride'，2則表示移動步長為2，因為與pool_size相同，所以2x2的塊就不會重複。
  - dropoutLayer(0.25)
這行添加了一個Dropout層，用於防止過擬合。rate=0.25意味著在訓練過程中，每個更新階段隨機丟棄25%的輸出單元。
  - model.add(layers.Flatten())
這行將前面層的輸出展平。即將多維的輸出轉化為一維，以便可以在後續的全連接層中使用。
  - fullyConnectedLayer(128)
添加了一個全連接（Dense）層，有128個神經元。
  - reluLayer()
添加了一個使用 Rectified Linear Unit (ReLU) 激活函数的層。
  - dropoutLayer(0.5)
再次添加了一個Dropout層，這次丟棄率為50%。
  - fullyConnectedLayer(numClasses)
添加了另一個全連接層，有36個神經元，對應於模型的輸出類別數量。
  - softmaxLayer()
添加Softmax激活函数的層，它常用於神經網絡的輸出層，特別是在進行多類別分類問題時。Softmax函數的作用主要就是將一組原始輸出值映射成一組表示概率分布的值，這組值的總和為1。每個輸出值代表了輸入樣本屬於特定類別的概率。
 - classificationLayer()
classificationLayer() 是 MATLAB 中用於指定網路架構最後一層的函数，其主要目的是進行分類任務。這層跟在softmax激活函數層之後，這在多類別分類問題中是常見的，用於將輸出轉化為概率分佈。

---
**三、深入了解一下這段程式中多次用到的ReLU的主要意義：**
 - 引入非線性： 
神經網絡中的非線性是非常重要的，因為大多數現實世界的數據是非線性的。
ReLU通過將所有負值轉化為0，同時保持正值不變，為網絡引入了非線性。
這種非線性使得網絡能夠學習並表示複雜的函數，從而處理各種複雜的數據。
 - 解決梯度消失問題：
在深度神經網絡中，梯度消失是一個常見問題，尤其是在使用傳統的激活函數（如sigmoid或tanh）時。
ReLU函數對於正輸入值有著恆定的梯度（即1），這有助於緩解在深層網絡中的梯度消失問題。
 - 計算效率高：
ReLU的計算非常簡單——只涉及閾值操作，這使得它比其他激活函數（如sigmoid或tanh）計算上更為高效。
這種高效率使得訓練深層網絡變得更加可行。
 - 促進稀疏激活：
在ReLU中，所有負值都被設為0，這意味著網絡的一部分神經元將會被“關閉”（即輸出為0）。
這種稀疏激活可以使模型更加高效和易於訓練，並有助於減少過擬合。
ReLU由於其簡單性和效率，在卷積神經網絡的設計中被廣泛使用，特別是在處理圖像識別和分類問題時。它的使用大幅改善了許多深度學習模型的性能和訓練速度。

---
- 訓練及測試模型，並輸出準確度
```matlab=
% Set Training Options
options = trainingOptions('adam', ...
    'InitialLearnRate', 0.001, ...
    'MaxEpochs', epochs, ...
    'MiniBatchSize', digitsInImg, ...
    'ValidationData', {cat(4, xTest{:}), yTest}, ...
    'ValidationFrequency', 30, ...
    'Verbose', true, ...
    'Plots', 'training-progress');
net = trainNetwork(cat(4, xTrain{:}), yTrain, layers, options);
% Evaluate Model
YPred = classify(net, cat(4, xTest{:}));
accuracy = sum(YPred == yTest) / numel(yTest);
fprintf('Test accuracy: %.2f%%\n', accuracy * 100);
```
- 將模型存檔
```matlab=
% Save Train Model
save('cnn_model.mat', 'net');
disp('New model created.');
```
- 定義分割圖片中的字元，並將圖像放進x_lis，檔名中的實際字元轉換的數字放進y_list
```matlab=
% Local function must be defined at the end
function [xList, yList] = splitDigitsInImg(imgArray, imgFilename, xList, yList, charToInt, digitsInImg)
    imgCols = size(imgArray, 2);
    step = imgCols / digitsInImg;
    for i = 1:digitsInImg
        slice = imgArray(:, round((i-1)*step+1):round(i*step));
        xList{end+1} = slice / 255;
        yList(end+1) = charToInt(imgFilename(i));
    end
end
```
#### 撰寫預測程式並載入訓練好的模型來對圖片進行辨識(test.m)

- 定義一些要使用的參數
```matlab=
digitsInImg = 4;
model = [];
```
- 建立數字對應到數字字元及大寫英文字元的對應表
```matlab=
% Create a reverse mapping from integers to characters
characters = ['0':'9', 'A':'Z'];  % This is a character array
characterCells = cellstr(characters');  % Convert character array to cell array of characters
intToChar = containers.Map(0:35, characterCells);  % Create map with integer keys and character cell array values
```
- 載入模型，不存在模型檔則報錯
```matlab=
% Check if the model exists
if exist('cnn_model.mat', 'file')
    loaded = load('cnn_model.mat');
    model = loaded.net;  % Assuming the network is saved under the name 'net'
else
    disp('No trained model found.');
    return;
end
```
- 讓使用者輸入要測試的圖片主檔名
```matlab=
imgFilename = input('Verification code img filename: ', 's');
```
- 載入圖檔
```matlab=
% Load an image
img = imread(['./testImgs/', imgFilename, '.jpg']);
img = rgb2gray(img);  % Convert to grayscale if necessary
imgArray = double(img) / 255;  % Normalize image
[imgRows, imgCols] = size(imgArray);
```
- 將分割的四個字元圖片放入x_list
```matlab=
% Split the image into digits
xList = {};
step = imgCols / digitsInImg;
for i = 1:digitsInImg
    xList{i} = imgArray(:, (i-1)*step+1:i*step);
end
```
- 預測x_list中的每個字元圖片的數字並轉換成對應的字元
```matlab=
% Predict using the loaded model
verificationCode = '';
for i = 1:digitsInImg
    confidences = predict(model, xList{i});
    [maxConfidence, resultClass] = max(confidences);
    predictedChar = intToChar(resultClass - 1);  % Adjust index for MATLAB
    verificationCode = [verificationCode, predictedChar];
    fprintf('Digit %d: Confidence=> %s    Predict=> %s\n', i, num2str(maxConfidence), predictedChar);
end
```
- 輸出預測的字元(CAPTCHA驗證碼)
```matlab=
fprintf('Predicted verification code: %s\n', verificationCode);
```
#### 爬取並下載目標網站中的CAPTCHA圖片辨識後存檔(更新後的fetch.m)
- 修改下載圖片並儲存的函式
```matlab=
% Function to download and save one CAPTCHA image
function success = downloadCaptcha(folderPath)
    options = weboptions('ContentType', 'image', 'Timeout', 30);
    captchaUrl = '此處使用動態生成CAPTCHA圖片的網址';
    try
        imdata = webread(captchaUrl, options);

        %randomCode = randi([0, 10000]);  % Generate a random integer between 0 and 10000
        %filePath = fullfile(folderPath, sprintf('%d.jpg', randomCode));
        
        %use predict instead of random
        code = predictCaptcha(imdata);
        filePath = fullfile(folderPath, sprintf('%s.jpg', code));

        imwrite(imdata, filePath);
        success = true;
    catch
        success = false;
    end
end
```
- 建立預測CAPTCHA驗證碼並回傳的函式
```matlab=
function code = predictCaptcha(img)
    model = [];  % MATLAB uses different formats, typically .mat for saved models
    digitsInImg = 4;
    % Create a reverse mapping from integers to characters
    characters = ['0':'9', 'A':'Z'];  % This is a character array
    characterCells = cellstr(characters');  % Convert character array to cell array of characters
    intToChar = containers.Map(0:35, characterCells);  % Create map with integer keys and character cell array values
    
    % Check if the model exists
    if exist('cnn_model.mat', 'file')
        loaded = load('cnn_model.mat');
        model = loaded.net;  % Assuming the network is saved under the name 'net'
    else
        disp('No trained model found.');
    end
    img = rgb2gray(img);  % Convert to grayscale if necessary
    imgArray = double(img) / 255;  % Normalize image
    [imgRows, imgCols] = size(imgArray);
    
    % Split the image into digits
    xList = {};
    step = imgCols / digitsInImg;
    for i = 1:digitsInImg
        xList{i} = imgArray(:, (i-1)*step+1:i*step);
    end
    
    % Predict using the loaded model
    verificationCode = '';
    for i = 1:digitsInImg
        confidences = predict(model, xList{i});
        [maxConfidence, resultClass] = max(confidences);
        predictedChar = intToChar(resultClass - 1);  % Adjust index for MATLAB
        verificationCode = [verificationCode, predictedChar];
        fprintf('Digit %d: Confidence=> %s    Predict=> %s\n', i, num2str(maxConfidence), predictedChar);
    end
    code = verificationCode;
end
```

#### 整合成自動登入網站的程式(login.m)
- 爬取驗證碼
```matlab=
% Retrieve CAPTCHA image
captcha_url = "此處使用動態生成CAPTCHA圖片的網址";
options = weboptions('ContentType', 'image');
img = webread(captcha_url, options);
```
- 呼叫預測CAPTCHA驗證碼並回傳的函式以取得預測的CAPTCHA驗證碼
```matlab=
% Predict Verification Code
verificationCode = predictCaptcha(img);
```
- 使用預測的CAPTCHA驗證碼登入網頁
```matlab=
% Define the URL for the login action and the data
loginUrl = '此處使用要登入的網頁';
postData = {'uid', '此處使用登入的帳號', 'pwd', '此處使用登入的密碼', 'checkCode', verificationCode};

% Create a web options object with MediaType set for form submission
options = weboptions('MediaType', 'application/x-www-form-urlencoded', 'Timeout', 10);

% Post the login data
response = webwrite(loginUrl, postData{:}, options);
```
- 取得登入後有權限造訪的網頁的結果
```matlab=
% Now use the same options to make further requests that require login session
afterLoginUrl = '此處使用登入後的網頁';
try
    pageContent = webread(afterLoginUrl, options);
    disp('Logged in successfully.');
    % disp(pageContent);
catch ME
    disp('Failed to access page content post-login');
    % disp(getReport(ME, 'extended'));
end
```
- 建立預測CAPTCHA驗證碼並回傳的函式
```matlab=
function code = predictCaptcha(img)
    model = [];  % MATLAB uses different formats, typically .mat for saved models
    digitsInImg = 4;
    % Create a reverse mapping from integers to characters
    characters = ['0':'9', 'A':'Z'];  % This is a character array
    characterCells = cellstr(characters');  % Convert character array to cell array of characters
    intToChar = containers.Map(0:35, characterCells);  % Create map with integer keys and character cell array values
    
    % Check if the model exists
    if exist('cnn_model.mat', 'file')
        loaded = load('cnn_model.mat');
        model = loaded.net;  % Assuming the network is saved under the name 'net'
    else
        disp('No trained model found.');
    end
    img = rgb2gray(img);  % Convert to grayscale if necessary
    imgArray = double(img) / 255;  % Normalize image
    [imgRows, imgCols] = size(imgArray);
    
    % Split the image into digits
    xList = {};
    step = imgCols / digitsInImg;
    for i = 1:digitsInImg
        xList{i} = imgArray(:, (i-1)*step+1:i*step);
    end
    
    % Predict using the loaded model
    verificationCode = '';
    for i = 1:digitsInImg
        confidences = predict(model, xList{i});
        [maxConfidence, resultClass] = max(confidences);
        predictedChar = intToChar(resultClass - 1);  % Adjust index for MATLAB
        verificationCode = [verificationCode, predictedChar];
        fprintf('Digit %d: Confidence=> %s    Predict=> %s\n', i, num2str(maxConfidence), predictedChar);
    end
    code = verificationCode;
end
```
**[參考資料]**
- 用TensorFlow+Keras訓練辨識驗證碼的CNN模型
https://notes.andywu.tw/2019/%E7%94%A8tensorflowkeras%E8%A8%93%E7%B7%B4%E8%BE%A8%E8%AD%98%E9%A9%97%E8%AD%89%E7%A2%BC%E7%9A%84cnn%E6%A8%A1%E5%9E%8B/
- TensorFlow:你也能成為機器學習專家 喻儼等著
https://nkust.ebook.hyread.com.tw/bookDetail.jsp?id=163664
- 卷積神經網路(Convolutional neural network, CNN) — CNN運算流程
https://chih-sheng-huang821.medium.com/%E5%8D%B7%E7%A9%8D%E7%A5%9E%E7%B6%93%E7%B6%B2%E8%B7%AF-convolutional-neural-network-cnn-cnn%E9%81%8B%E7%AE%97%E6%B5%81%E7%A8%8B-ecaec240a631
- 别再在CNN中使用Dropout了
https://imgtec.eetrend.com/blog/2021/100555349.html
