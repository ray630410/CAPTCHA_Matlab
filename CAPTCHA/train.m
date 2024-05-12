% Initialize Variables
epochs = 10;       % Number of epochs for training
digitsInImg = 4;   % CAPTCHA length
numClasses = 36;   % 26 English alphabets + 10 digits
xList = {};        % Store image data
yList = [];        % Store character

% Create map from characters to integers
characters = ['0':'9', 'A':'Z'];  % Digits + uppercase letters
charToInt = containers.Map('KeyType', 'char', 'ValueType', 'int32');
for i = 1:length(characters)
    charToInt(characters(i)) = i;
end

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

% Convert Labels to Categorical
yList = categorical(yList, 1:numClasses);

% Split Data into Training and Testing Sets
perm = randperm(numel(xList));
idxTrain = perm(1:round(0.9*numel(xList)));
idxTest = perm(round(0.9*numel(xList))+1:end);
xTrain = xList(idxTrain);
yTrain = yList(idxTrain);
xTest = xList(idxTest);
yTest = yList(idxTest);

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

% Save Train Model
save('cnn_model.mat', 'net');
disp('New model created.');

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
