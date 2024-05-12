% Set up the environment
folderPath = './download';
if ~exist(folderPath, 'dir')
    mkdir(folderPath);
end

% Download and save 10 images
successCount = 0;
for k = 1:10  % Adjust the range if you want more or fewer images
    if downloadCaptcha(folderPath)
        successCount = successCount + 1;
    end
    pause(3);  % Wait for 3 seconds before downloading the next image
end

disp(['Successfully downloaded ' num2str(successCount) ' images.']);

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