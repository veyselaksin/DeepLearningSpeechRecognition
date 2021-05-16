%% Speech Command Recognition Using Deep Learning
% This example shows how to train a deep learning model that detects the
% presence of speech commands in audio. The example uses the Speech
% Commands Dataset [1] to train a convolutional neural network to recognize
% a given set of commands.
%
% To train a network from scratch, you must first download the data set. If
% you do not want to download the data set or train the network, then you
% can load a pretrained network provided with this example and execute the
% next two sections of the example: _Recognize Commands with a Pre-Trained
% Network_ and _Detect Commands Using Streaming Audio from Microphone_.

%% Recognize Commands with a Pre-Trained Network
% Before going into the training process in detail, you will use a
% pre-trained speech recognition network to identify speech commands.

%%
% Load the pre-trained network.
load('commandNet.mat')
 
%%
% The network is trained to recognize the following speech commands:
%
% * "yes"
% * "no"
% * "up"
% * "down"
% * "left"
% * "right"
% * "on"
% * "off"
% * "stop"
% * "go"
%

%%
% Load a short speech signal where a person says "stop".
 [x,fs] = audioread('stop_command.flac');
 
 %%
 % Listen to the command.
 sound(x,fs)
 
%%
% The pre-trained network takes auditory-based spectrograms as inputs. You
% will first convert the speech waveform to an auditory-based spectrogram.

%%
% Use the function |extractAuditoryFeature| to compute the auditory
% spectrogram. You will go through the details of feature extraction later
% in the example.
auditorySpect = helperExtractAuditoryFeatures(x,fs);

%%
% Classify the command based on its auditory spectrogram.
command = classify(trainedNet,auditorySpect)

%%
% The network is trained to classify words not belonging to this set as
% "unknown".
% 
% You will now classify a word ("play") that was not included in the list
% of command to identify. 

%%
% Load the speech signal and listen to it.
x = audioread('play_command.flac');
sound(x,fs)

%%
% Compute the auditory spectrogram.
auditorySpect = helperExtractAuditoryFeatures(x,fs);

%%
% Classify the signal.
command = classify(trainedNet,auditorySpect)

%%
% The network is trained to classify background noise as "background".

%%
% Create a one-second signal consisting of random noise.
x = pinknoise(16e3);

%%
% Compute the auditory spectrogram.
auditorySpect = helperExtractAuditoryFeatures(x,fs);

%%
% Classify the background noise.
command = classify(trainedNet,auditorySpect)

%% Detect Commands Using Streaming Audio from Microphone
% Test your pre-trained command detection network on streaming audio from
% your microphone. Try saying one of the commands, for example, _yes_,
% _no_, or _stop_. Then, try saying one of the unknown words such as
% _Marvin_, _Sheila_, _bed_, _house_, _cat_, _bird_, or any number from
% zero to nine.
%
% Specify the classification rate in Hz and create an audio device reader
% that can read audio from your microphone.

classificationRate = 20;
adr = audioDeviceReader('SampleRate',fs,'SamplesPerFrame',floor(fs/classificationRate));

%%
% Initialize a buffer for the audio. Extract the classification labels of
% the network. Initialize buffers of half a second for the labels and
% classification probabilities of the streaming audio. Use these buffers to
% compare the classification results over a longer period of time and by
% that build 'agreement' over when a command is detected. Specify
% thresholds for the decision logic.

audioBuffer = dsp.AsyncBuffer(fs);

labels = trainedNet.Layers(end).Classes;
YBuffer(1:classificationRate/2) = categorical("background");

probBuffer = zeros([numel(labels),classificationRate/2]);

countThreshold = ceil(classificationRate*0.2);
probThreshold = 0.7;

%%
% Create a figure and detect commands as long as the created figure exists.
% To run the loop indefinitely, set |timeLimit| to |Inf|. To stop the live
% detection, simply close the figure.

h = figure('Units','normalized','Position',[0.2 0.1 0.6 0.8]);

timeLimit = 20;

tic
while ishandle(h) && toc < timeLimit
    
    % Extract audio samples from the audio device and add the samples to
    % the buffer.
    x = adr();
    write(audioBuffer,x);
    y = read(audioBuffer,fs,fs-adr.SamplesPerFrame);
    
    spec = helperExtractAuditoryFeatures(y,fs);
    
    % Classify the current spectrogram, save the label to the label buffer,
    % and save the predicted probabilities to the probability buffer.
    [YPredicted,probs] = classify(trainedNet,spec,'ExecutionEnvironment','cpu');
    YBuffer = [YBuffer(2:end),YPredicted];
    probBuffer = [probBuffer(:,2:end),probs(:)];
    
    % Plot the current waveform and spectrogram.
    subplot(2,1,1)
    plot(y)
    axis tight
    ylim([-1,1])
    
    subplot(2,1,2)
    pcolor(spec')
    caxis([-4 2.6445])
    shading flat
    
    % Now do the actual command detection by performing a very simple
    % thresholding operation. Declare a detection and display it in the
    % figure title if all of the following hold: 1) The most common label
    % is not background. 2) At least countThreshold of the latest frame
    % labels agree. 3) The maximum probability of the predicted label is at
    % least probThreshold. Otherwise, do not declare a detection.
    [YMode,count] = mode(YBuffer);
    
    maxProb = max(probBuffer(labels == YMode,:));
    subplot(2,1,1)
    if YMode == "background" || count < countThreshold || maxProb < probThreshold
        title(" ")
    else
        title(string(YMode),'FontSize',20)
    end
    
    drawnow
end

%%
% <<../streaming_commands.png>>



%% Visualize Data
% Plot the waveforms and auditory spectrograms of a few training samples.
% Play the corresponding audio clips.

specMin = min(XTrain,[],'all');
specMax = max(XTrain,[],'all');
idx = randperm(numel(adsTrain.Files),3);
figure('Units','normalized','Position',[0.2 0.2 0.6 0.6]);
for i = 1:3
    [x,fs] = audioread(adsTrain.Files{idx(i)});
    subplot(2,3,i)
    plot(x)
    axis tight
    title(string(adsTrain.Labels(idx(i))))
    
    subplot(2,3,i+3)
    spect = (XTrain(:,:,1,idx(i))');
    pcolor(spect)
    caxis([specMin specMax])
    shading flat
    
    sound(x,fs)
    pause(2)
end

%%
% Plot the distribution of the different class labels in the training and
% validation sets.

figure('Units','normalized','Position',[0.2 0.2 0.5 0.5])

subplot(2,1,1)
histogram(YTrain)
title("Training Label Distribution")

subplot(2,1,2)
histogram(YValidation)
title("Validation Label Distribution")


%% Evaluate Trained Network
% Calculate the final accuracy of the network on the training set (without
% data augmentation) and validation set. The network is very accurate on
% this data set. However, the training, validation, and test data all have
% similar distributions that do not necessarily reflect real-world
% environments. This limitation particularly applies to the |unknown|
% category, which contains utterances of only a small number of words.
if reduceDataset
    load('commandNet.mat','trainedNet');
end
YValPred = classify(trainedNet,XValidation);
validationError = mean(YValPred ~= YValidation);
YTrainPred = classify(trainedNet,XTrain);
trainError = mean(YTrainPred ~= YTrain);
disp("Training error: " + trainError*100 + "%")
disp("Validation error: " + validationError*100 + "%")

%%
% Plot the confusion matrix. Display the precision and recall for each
% class by using column and row summaries. Sort the classes of the
% confusion matrix. The largest confusion is between unknown words and
% commands, _up_ and _off_, _down_ and _no_, and _go_ and _no_.

figure('Units','normalized','Position',[0.2 0.2 0.5 0.5]);
cm = confusionchart(YValidation,YValPred);
cm.Title = 'Confusion Matrix for Validation Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
sortClasses(cm, [commands,"unknown","background"])

%%
% When working on applications with constrained hardware resources such as
% mobile applications, consider the limitations on available memory and
% computational resources. Compute the total size of the network in
% kilobytes and test its prediction speed when using a CPU. The prediction
% time is the time for classifying a single input image. If you input
% multiple images to the network, these can be classified simultaneously,
% leading to shorter prediction times per image. When classifying streaming
% audio, however, the single-image prediction time is the most relevant.

info = whos('trainedNet');
disp("Network size: " + info.bytes/1024 + " kB")

for i = 1:100
    x = randn([numHops,numBands]);
    tic
    [YPredicted,probs] = classify(trainedNet,x,"ExecutionEnvironment",'cpu');
    time(i) = toc;
end
disp("Single-image prediction time on CPU: " + mean(time(11:end))*1000 + " ms")

%% References
% [1] Warden P. "Speech Commands: A public dataset for single-word speech
% recognition", 2017. Available from
% https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.01.tar.gz.
% Copyright Google 2017. The Speech Commands Dataset is licensed under the
% Creative Commons Attribution 4.0 license, available here:
% https://creativecommons.org/licenses/by/4.0/legalcode.

%%
%
% Copyright 2018 The MathWorks, Inc.