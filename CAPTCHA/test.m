% Add necessary toolboxes and setup paths (if not already set)
% addpath(genpath('Deep Learning Toolbox'));

% Initialize variables
digitsInImg = 4;
model = [];

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
    return;
end

imgFilename = input('Verification code img filename: ', 's');

% Load an image
img = imread(['./testImgs/', imgFilename, '.jpg']);
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

fprintf('Predicted verification code: %s\n', verificationCode);
