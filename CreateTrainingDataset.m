%% Load Speech Commands Data Set
% This example uses the Google Speech Commands Dataset [1]. Download the 
% dataset and untar the downloaded file. Set PathToDatabase to the location 
% of the data.
dataFolder = fullfile('google_speech');

%% Create Training Datastore
% Create an <docid:audio_ref#mw_6315b106-9a7b-4a11-a7c6-322c073e343a
% audioDatastore> that points to the training data set.
ads = audioDatastore(fullfile(dataFolder, 'train'), ...
    'IncludeSubfolders',true, ...
    'FileExtensions','.wav', ...
    'LabelSource','foldernames')

%% Choose Words to Recognize
% Specify the words that you want your model to recognize as commands.
% Label all words that are not commands as |unknown|. Labeling words that
% are not commands as |unknown| creates a group of words that approximates
% the distribution of all words other than the commands. The network uses
% this group to learn the difference between commands and all other words.
%
% To reduce the class imbalance between the known and unknown words and
% speed up processing, only include a fraction of the unknown words in the
% training set. 
%
% Use <docid:audio_ref#mw_6823f1d7-3610-4d7d-89d0-816746a24ca9 subset> to
% create a datastore that contains only the commands and the subset of
% unknown words. Count the number of examples belonging to each category.

commands = categorical(["bed","bird","cat","dog","down","go","happy","house","left","marvin","no","off","on","right","sheila","stop","tree","up","yes","zero","one","two","three","four","five"]);

isCommand = ismember(ads.Labels,commands);
isUnknown = ~isCommand;

includeFraction = 0.2;
mask = rand(numel(ads.Labels),1) < includeFraction;
isUnknown = isUnknown & mask;
ads.Labels(isUnknown) = categorical("unknown");

adsTrain = subset(ads,isCommand|isUnknown);
countEachLabel(adsTrain)

%% Create Validation Datastore
% Create an <docid:audio_ref#mw_6315b106-9a7b-4a11-a7c6-322c073e343a
% audioDatastore> that points to the validation data set. Follow the same
% steps used to create the training datastore.
ads = audioDatastore(fullfile(dataFolder, 'validation'), ...
    'IncludeSubfolders',true, ...
    'FileExtensions','.wav', ...
    'LabelSource','foldernames')

isCommand = ismember(ads.Labels,commands);
isUnknown = ~isCommand;

includeFraction = 0.2;
mask = rand(numel(ads.Labels),1) < includeFraction;
isUnknown = isUnknown & mask;
ads.Labels(isUnknown) = categorical("unknown");

adsValidation = subset(ads,isCommand|isUnknown);
countEachLabel(adsValidation)

%%
% To train the network with the entire dataset and achieve the highest
% possible accuracy, set |reduceDataset| to |false|. To run this example
% quickly, set |reduceDataset| to |true|.
reduceDataset = false;
if reduceDataset
    numUniqueLabels = numel(unique(adsTrain.Labels));
    % Reduce the dataset by a factor of 20
    adsTrain = splitEachLabel(adsTrain,round(numel(adsTrain.Files) / numUniqueLabels / 20));
    adsValidation = splitEachLabel(adsValidation,round(numel(adsValidation.Files) / numUniqueLabels / 20));
end

%% Compute Auditory Spectrograms
% To prepare the data for efficient training of a convolutional neural
% network, convert the speech waveforms to auditory-based spectrograms.
%
% Define the parameters of the feature extraction. |segmentDuration| is the
% duration of each speech clip (in seconds). |frameDuration| is the
% duration of each frame for spectrum calculation. |hopDuration| is the
% time step between each spectrum. |numBands| is the number of filters
% in the auditory spectrogram.
%
% Create an <docid:audio_ref#mw_b56cd7dc-af31-4da4-a43e-b13debc30322
% audioFeatureExtractor> object to perform the feature extraction.

fs = 16e3; % Known sample rate of the data set.

segmentDuration = 1;
frameDuration = 0.025;
hopDuration = 0.010;

segmentSamples = round(segmentDuration*fs);
frameSamples = round(frameDuration*fs);
hopSamples = round(hopDuration*fs);
overlapSamples = frameSamples - hopSamples;

FFTLength = 512;
numBands = 50;

afe = audioFeatureExtractor( ...
    'SampleRate',fs, ...
    'FFTLength',FFTLength, ...
    'Window',hann(frameSamples,'periodic'), ...
    'OverlapLength',overlapSamples, ...
    'barkSpectrum',true);
setExtractorParams(afe,'barkSpectrum','NumBands',numBands,'WindowNormalization',false);

%%
% Read a file from the dataset. Training a convolutional neural network
% requires input to be a consistent size. Some files in the data set are
% less than 1 second long. Apply zero-padding to the front and back of
% the audio signal so that it is of length |segmentSamples|.
x = read(adsTrain);

numSamples = size(x,1);

numToPadFront = floor( (segmentSamples - numSamples)/2 );
numToPadBack = ceil( (segmentSamples - numSamples)/2 );

xPadded = [zeros(numToPadFront,1,'like',x);x;zeros(numToPadBack,1,'like',x)];

%%
% To extract audio features, call |extract|. The output is a Bark spectrum
% with time across rows.
features = extract(afe,xPadded);
[numHops,numFeatures] = size(features)

%%
% In this example, you post-process the auditory
% spectrogram by applying a logarithm. Taking a log of small numbers can
% lead to roundoff error.
%

%%
% To speed up processing, you can distribute the feature extraction across
% multiple workers using |parfor|. 
%
% First, determine the number of partitions for the dataset. If you do not
% have Parallel Computing Toolbox(TM), use a single partition.
if ~isempty(ver('parallel')) && ~reduceDataset
    pool = gcp;
    numPar = numpartitions(adsTrain,pool);
else
    numPar = 1;
end

%%
% For each partition, read from the datastore, zero-pad the signal, and
% then extract the features.

parfor ii = 1:numPar
    subds = partition(adsTrain,numPar,ii);
    XTrain = zeros(numHops,numBands,1,numel(subds.Files));
    for idx = 1:numel(subds.Files)
        x = read(subds);
        xPadded = [zeros(floor((segmentSamples-size(x,1))/2),1);x;zeros(ceil((segmentSamples-size(x,1))/2),1)];
        XTrain(:,:,:,idx) = extract(afe,xPadded);
    end
    XTrainC{ii} = XTrain;
end

%%
% Convert the output to a 4-dimensional array with auditory spectrograms
% along the fourth dimension.

XTrain = cat(4,XTrainC{:});

[numHops,numBands,numChannels,numSpec] = size(XTrain)

%%
% Scale the features by the window power and then take the log. To obtain
% data with a smoother distribution, take the logarithm of the spectrograms
% using a small offset.

epsil = 1e-6;
XTrain = log10(XTrain + epsil);

%%
% Perform the feature extraction steps described above to the validation
% set.

if ~isempty(ver('parallel'))
    pool = gcp;
    numPar = numpartitions(adsValidation,pool);
else
    numPar = 1;
end
parfor ii = 1:numPar
    subds = partition(adsValidation,numPar,ii);
    XValidation = zeros(numHops,numBands,1,numel(subds.Files));
    for idx = 1:numel(subds.Files)
        x = read(subds);
        xPadded = [zeros(floor((segmentSamples-size(x,1))/2),1);x;zeros(ceil((segmentSamples-size(x,1))/2),1)];
        XValidation(:,:,:,idx) = extract(afe,xPadded);
    end
    XValidationC{ii} = XValidation;
end
XValidation = cat(4,XValidationC{:});
XValidation = log10(XValidation + epsil);

%%
% Isolate the train and validation labels. Remove empty categories.

YTrain = removecats(adsTrain.Labels);
YValidation = removecats(adsValidation.Labels);
%% Add Background Noise Data
% The network must be able not only to recognize different spoken words but
% also to detect if the input contains silence or background noise.
%
% Use the audio files in the |_background|_ folder to create samples
% of one-second clips of background noise. Create an equal number of
% background clips from each background noise file. You can also create
% your own recordings of background noise and add them to the
% |_background|_ folder. Before calculating the spectrograms, the
% function rescales each audio clip with a factor sampled from a
% log-uniform distribution in the range given by |volumeRange|.

adsBkg = audioDatastore(fullfile(dataFolder, 'background'))
numBkgClips = 4000;
if reduceDataset
    numBkgClips = numBkgClips/20;
end
volumeRange = log10([1e-4,1]);

numBkgFiles = numel(adsBkg.Files);
numClipsPerFile = histcounts(1:numBkgClips,linspace(1,numBkgClips,numBkgFiles+1));
Xbkg = zeros(size(XTrain,1),size(XTrain,2),1,numBkgClips,'single');
bkgAll = readall(adsBkg);
ind = 1;

for count = 1:numBkgFiles
    bkg = bkgAll{count};
    idxStart = randi(numel(bkg)-fs,numClipsPerFile(count),1);
    idxEnd = idxStart+fs-1;
    gain = 10.^((volumeRange(2)-volumeRange(1))*rand(numClipsPerFile(count),1) + volumeRange(1));
    for j = 1:numClipsPerFile(count)
        
        x = bkg(idxStart(j):idxEnd(j))*gain(j);
        
        x = max(min(x,1),-1);
        
        Xbkg(:,:,:,ind) = extract(afe,x);
        
        if mod(ind,1000)==0
            disp("Processed " + string(ind) + " background clips out of " + string(numBkgClips))
        end
        ind = ind + 1;
    end
end
Xbkg = log10(Xbkg + epsil);

%%
% Split the spectrograms of background noise between the training,
% validation, and test sets. Because the |_background_noise|_ folder
% contains only about five and a half minutes of background noise, the
% background samples in the different data sets are highly correlated. To
% increase the variation in the background noise, you can create your own
% background files and add them to the folder. To increase the robustness
% of the network to noise, you can also try mixing background noise into
% the speech files.

numTrainBkg = floor(0.85*numBkgClips);
numValidationBkg = floor(0.15*numBkgClips);

XTrain(:,:,:,end+1:end+numTrainBkg) = Xbkg(:,:,:,1:numTrainBkg);
YTrain(end+1:end+numTrainBkg) = "background";

XValidation(:,:,:,end+1:end+numValidationBkg) = Xbkg(:,:,:,numTrainBkg+1:end);
YValidation(end+1:end+numValidationBkg) = "background";


%% Define Neural Network Architecture
% Create a simple network architecture as an array of layers. Use
% convolutional and batch normalization layers, and downsample the feature
% maps "spatially" (that is, in time and frequency) using max pooling
% layers. Add a final max pooling layer that pools the input feature map
% globally over time. This enforces (approximate) time-translation
% invariance in the input spectrograms, allowing the network to perform the
% same classification independent of the exact position of the speech in
% time. Global pooling also significantly reduces the number of parameters
% in the final fully connected layer. To reduce the possibility of the
% network memorizing specific features of the training data, add a small
% amount of dropout to the input to the last fully connected layer.
%
% The network is small, as it has only five convolutional layers with few
% filters. |numF| controls the number of filters in the convolutional
% layers. To increase the accuracy of the network, try increasing the
% network depth by adding identical blocks of convolutional, batch
% normalization, and ReLU layers. You can also try increasing the number of
% convolutional filters by increasing |numF|.
%
% Use a weighted cross entropy classification loss.
% <matlab:edit(fullfile(matlabroot,'examples','deeplearning_shared','main','weightedClassificationLayer.m'))
% |weightedClassificationLayer(classWeights)|> creates a custom
% classification layer that calculates the cross entropy loss with
% observations weighted by |classWeights|. Specify the class weights in the
% same order as the classes appear in |categories(YTrain)|. To give each
% class equal total weight in the loss, use class weights that are
% inversely proportional to the number of training examples in each class.
% When using the Adam optimizer to train the network, the training
% algorithm is independent of the overall normalization of the class
% weights.

classWeights = 1./countcats(YTrain);
classWeights = classWeights'/mean(classWeights);
numClasses = numel(categories(YTrain));

timePoolSize = ceil(numHops/8);

dropoutProb = 0.2;
numF = 12;
layers = [
    imageInputLayer([numHops numBands])
    
    convolution2dLayer(3,numF,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(3,'Stride',2,'Padding','same')
    
    convolution2dLayer(3,2*numF,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(3,'Stride',2,'Padding','same')
    
    convolution2dLayer(3,4*numF,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(3,'Stride',2,'Padding','same')
    
    convolution2dLayer(3,4*numF,'Padding','same')
    batchNormalizationLayer
    reluLayer
    convolution2dLayer(3,4*numF,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer([timePoolSize,1])
    
    dropoutLayer(dropoutProb)
    fullyConnectedLayer(numClasses)
    softmaxLayer
    weightedClassificationLayer(classWeights)];

%% Train Network
% Specify the training options. Use the Adam optimizer with a mini-batch
% size of 128. Train for 25 epochs and reduce the learning rate by a factor
% of 10 after 20 epochs.

miniBatchSize = 128;
validationFrequency = floor(numel(YTrain)/miniBatchSize);
options = trainingOptions('adam', ...
    'InitialLearnRate',3e-4, ...
    'MaxEpochs',25, ...
    'MiniBatchSize',miniBatchSize, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...
    'Verbose',false, ...
    'ValidationData',{XValidation,YValidation}, ...
    'ValidationFrequency',validationFrequency, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.1, ...
    'LearnRateDropPeriod',20);

%%
% Train the network. If you do not have a GPU, then training the network
% can take time.
trainedNet = trainNetwork(XTrain,YTrain,layers,options);

