clc;
clear all;
close all;

%% reading image files from the specified folder

myfolder = 'D:\NIT CONFERENCE WORK 2020\chest_xray\train 1000 512';

DatasetPath = fullfile(myfolder);
imds = imageDatastore(DatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames','FileExtensions',{'.jpg','.png', '.jpeg'});


%% Specify Training and Validation Sets
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.8,'randomized');

%% code for resizing
pixelrange = [-30 30];
inputSize = [512 512 1];
aug = imageDataAugmenter('RandRotation',[-20,20], ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelrange, ...
    'RandYTranslation',pixelrange);
augimdsTrain = augmentedImageDatastore(inputSize,imdsTrain, ...
    'DataAugmentation',aug, ...
    'ColorPreprocessing','rgb2gray');
augimdsValidation = augmentedImageDatastore(inputSize,imdsValidation, ...
    'DataAugmentation',aug,...
    'ColorPreprocessing','rgb2gray');

%% label display
labelCount = countEachLabel(imds)


%% Define Network Architecture

layers = [
    imageInputLayer(inputSize,'Name','input')
    
    convolution2dLayer(3,32,'Padding',1,'Name','conv_1')
    batchNormalizationLayer('Name','BN1')
    reluLayer('Name','relu1')
    
    maxPooling2dLayer(2,'Stride',2,'Name','maxPool_1')
    
    convolution2dLayer(3,16,'Padding',1,'Name','conv_2')
    batchNormalizationLayer('Name','BN2')
    reluLayer('Name','relu2')
    
    maxPooling2dLayer(2,'Stride',2,'Name','maxPool_2')
    
    convolution2dLayer(3,8,'Padding',1,'Name','conv_3')
    batchNormalizationLayer('Name','BN3')
    reluLayer('Name','relu3')
    
    maxPooling2dLayer(2,'Stride',2,'Name','maxPool_3')
    
    
    fullyConnectedLayer(2,'Name', 'fullyConnected')
    softmaxLayer('Name','softmax')
    classificationLayer('Name', 'classify')];

%% Specify Training Options
options = trainingOptions('sgdm', ...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropFactor',0.2,...
    'LearnRateDropPeriod',5,...
    'MiniBatchSize',10, ...                                
    'ExecutionEnvironment','cpu', ...
    'MaxEpochs',10, ...
    'InitialLearnRate',1e-3, ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',5, ...
    'ValidationPatience',Inf, ...
    'Verbose',1 ,...
    'VerboseFrequency',10,...
    'Plots','training-progress');
%% Train Network Using Training Data
net = trainNetwork(augimdsTrain,layers,options);


%% layer graphs

lgraph = layerGraph(layers);
figure
plot(lgraph)


%% TEST
[YPred,probs] = classify(net,augimdsValidation, 'ExecutionEnvironment','cpu');
idx = randperm(numel(imdsValidation.Files),4);
figure
for i = 1:4
    subplot(2,2,i)
    I = readimage(imdsValidation,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label) + ", " + num2str(100*max(probs(idx(i),:)),3) + "%");
end

 
 %% Confusion matrix
 YValidation = classify(net, augimdsValidation,'ExecutionEnvironment','cpu');
plotconfusion(YValidation,YPred);





