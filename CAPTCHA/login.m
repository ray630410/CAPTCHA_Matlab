% Retrieve CAPTCHA image
captcha_url = "此處使用動態生成CAPTCHA圖片的網址";
options = weboptions('ContentType', 'image');
img = webread(captcha_url, options);

% Predict Verification Code
verificationCode = predictCaptcha(img);

% Define the URL for the login action and the data
loginUrl = '此處使用要登入的網頁';
postData = {'uid', '此處使用登入的帳號', 'pwd', '此處使用登入的密碼', 'checkCode', verificationCode};

% Create a web options object with MediaType set for form submission
options = weboptions('MediaType', 'application/x-www-form-urlencoded', 'Timeout', 10);

% Post the login data
response = webwrite(loginUrl, postData{:}, options);

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
