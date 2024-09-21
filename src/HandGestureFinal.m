clear all

data = csvread('/Users/anabellehassan/Documents/BiomedicalEngineer/3/Semestre 2/Rehab/Project/HandGestureData.csv', 1, 0);
header = strsplit(fgetl(fopen('/Users/anabellehassan/Docu!ments/BiomedicalEngineer/3/Semestre 2/Rehab/Project/HandGestureData.csv')), ',');

%% Raw data plot (Electrode 9)
%plot of raw data (Electrode 9)
figure;
subplot(211);
plot(data(:,28),data(:,10));
%xlim([0 20]);
xlim([0 265]);
title('Electrode 9, Before data removal');
xlabel('Time(s)');
ylabel('Raw sEMG(mV)');
subplot(212)
plot(data(:,28),data(:,29));
%xlim([0 20]);
xlim([0 265]);
title('Handgesture');
xlabel('Time(s)');
ylabel('Hand Gesture');
hold off 

%% removing other two bands of electrodes
%removing band 1 and 3 of electrodes and beginning to focus on the middle 8
 data(:,[2:9,18:25])=[];
 
%%
 % Find indices where the value is -1
relaxindices = find(data(:,13) == -1);
data(relaxindices,:)=[]; 

%% plot without relaxed state (Electrode 9)
%Plot of data without -1 gesture
figure;
subplot(211);
plot(data(:,12),data(:,2));
%xlim([0 20]);
xlim([0 265]);
title('Electrode 9, After data removal');
xlabel('Time(s)');
ylabel('Raw sEMG(mV)');
subplot(212)
plot(data(:,12),data(:,13));
%xlim([0 20]);
xlim([0 265]);
title('Handgesture');
xlabel('Time(s)');
ylabel('Hand Gestures');
hold off 

%% plotting the first 50 seconds of the data for each of the selected electrodes (9-16)
% for i=2:9
%     figure;
%     subplot(211)
%     plot(data(:,12),data(:,i),'color','blue');
%     xlim([0 50]);
%     title('Electrode:', num2str(i+7));
%     xlabel('Time(s)');
%     ylabel('Raw sEMG(mV)');
%     hold on 
% 
%     subplot(212)
% 
% 
%     plot(data(:,12),data(:,13));
%     xlim([0 50]);
%     title('Handgesture');
%     xlabel('Time(s)');
% hold off
% end 

%% find gesture changeover points
gesture_data=data(:,13);
points_to_delete = [];
% Iterate over the gesture data
for i = 2:length(gesture_data)
    if gesture_data(i) ~= gesture_data(i-1)
        if gesture_data(i) ~= 0 || gesture_data(i-1) ~= 0
            start_index = max(1, i - 1280);
            end_index = min(length(gesture_data), i + 1280);
            points_to_delete = [points_to_delete, start_index:end_index];
        end
    end
end
cleaned_gesture_data = gesture_data;
cleaned_gesture_data(points_to_delete) = [];
cleaned_emg_data = data;
cleaned_emg_data(points_to_delete, :) = [];

%% without movemenet artefacts (Electrode 9)
%without artefacts
figure;
subplot(311);
plot(data(:,12),data(:,2));
%xlim([0 20]);
xlim([0 265]);
title('Electrode 9, After data removal');
xlabel('Time(s)');
ylabel('Raw sEMG(mV)');
subplot(312)
plot(cleaned_emg_data(:,12),cleaned_emg_data(:,2));
%xlim([0 20]);
xlim([0 265]);
title('Electrode 9, After artefact removal');
xlabel('Time(s)');
ylabel('Raw sEMG(mV)');
subplot(313)
plot(data(:,12),data(:,13));
%xlim([0 20]);
xlim([0 265]);
title('Handgesture');
xlabel('Time(s)');
ylabel('Hand Gestures');
hold off 

%% bandpass filter 
%Apply bandpass filter 
fs = 5120; 
lowcutoff = 20;   
highcutoff = 700; 
[b, a] = butter(4, [lowcutoff/(fs/2), highcutoff/(fs/2)], 'bandpass');
bandfiltered_emg_data = filtfilt(b, a, cleaned_emg_data(:,2:9));

%% notch filter attenuating frequencies 
frequencies_to_attenuate = [30, 50, 60, 90, 150]; 
bandwidth = 1; 
filtered_emg_data = bandfiltered_emg_data;
% Apply adaptive notch filter for each frequency
for f = frequencies_to_attenuate
    wo = f / (fs/2); 
    bw = bandwidth / (fs/2); 
    [b, a] = iirnotch(wo, bw);
    filtered_emg_data = filtfilt(b, a, filtered_emg_data);
end
% Plot the original and notch filtered EMG data and bandpass 
t = (0:length(cleaned_emg_data)-1) / fs; % Time vector
figure;
subplot(3,1,1);
plot(t, cleaned_emg_data(:,2));
title('Original EMG Data');
xlabel('Time (s)');
ylabel('Raw sEMG(mV)');
subplot(3,1,2);
plot(t, bandfiltered_emg_data(:,2));
title('Filtered EMG Data (Bandpass 20-700 Hz)');
xlabel('Time (s)');
ylabel('Raw sEMG(mV)');
subplot(3,1,3);
plot(t, filtered_emg_data(:,2));
title('Filtered EMG Data (Adaptive Notch Filter)');
xlabel('Time (s)');
ylabel('Raw sEMG(mV)');

%% add gesture data back to data 
gestures=cleaned_emg_data(:,13);
filtered_emg_data(:,9)=gestures;

%% feature extraction 
% identify a period with a steady force and steady burst of EMG activity
load('DataDetails.mat')
%first idle values 
% E9I1=E9(1:15360);
% E9I1=sqrt(mean(E9I1.^2));
E9=filtered_emg_data(:,1); %levels 5 and 6 or 7
E10=filtered_emg_data(:,2); %levels 5 and 6 or 7
E11=filtered_emg_data(:,3); %levels 5 and 6 or 7
E12=filtered_emg_data(:,4); %level 5 or 6
E13=filtered_emg_data(:,5); %levels 5 and 6 or 7
E14=filtered_emg_data(:,6); %levels 5 and 6 or 7
E15=filtered_emg_data(:,7); %levels 5 and 6 or 7
E16=filtered_emg_data(:,8); %levels 6 or 7 

%% Feature extraction
 
% FFT to find frequencies and decide levels we are using. 
% To find FFT, we use SignalAnalyzer
 
E9=cleaned_emg_data(:,2);
% E9: top 50% is 20 to 120 Hz. That corresponds to levels L5, L6 and L7.
% We export L5 and L6 from the wavelet analyzer. L7 is too small. 
 
% E10=cleaned_emg_data(:,3);
% E10: top 50% is 

%%  Wavelet analysis
 
% WaveletAnalyzer 
% we need to extract the corresponding decomposition levels from each electrode
 
% DataDetails(:,1)=E9det(5,:); %E9 level 5
% DataDetails(:,2)=E9det(6,:); %E9 level 6
% DataDetails(:,3)=E10det(5,:); %E10 level 5
% DataDetails(:,4)=E10det(6,:); %E10 level 6
% DataDetails(:,5)=E11det(5,:); %E11 level 5
% DataDetails(:,6)=E11det(6,:); %E11 level 6
% DataDetails(:,7)=E12det(5,:); %E12 level 5
% DataDetails(:,8)=E12det(6,:); %E12 level 6
% DataDetails(:,9)=E13det(5,:); %E13 level 5
% DataDetails(:,10)=E13det(6,:); %E13 level 6
% DataDetails(:,11)=E14det(5,:); %E10 level 5
% DataDetails(:,12)=E14det(6,:); %E10 level 6
% DataDetails(:,13)=E15det(5,:); %E10 level 5
% DataDetails(:,14)=E15det(6,:); %E10 level 6
% DataDetails(:,15)=E16det(6,:); %E10 level 6
% DataDetails(:,16)=E16det(7,:); %E10 level 7
 
% we just use DataDetails from now on (saved in workspace)

%% Add Gestures to datadetails
load('DataDetails.mat');
DataDetails(:,17)=filtered_emg_data(:,9);

%% rms 
windowLength = 300; 
stepSize = 50; 
[numRows, numColumns] = size(DataDetails);
numSegments = floor((numRows - windowLength) / stepSize) + 1;
rmsValues = zeros(numSegments, numColumns);
middleValues = zeros(numSegments, numColumns); 
for i = 1:numColumns
    for j = 1:numSegments
        startIndex = (j-1) * stepSize + 1;
        endIndex = startIndex + windowLength - 1;
        if endIndex > numRows 
            endIndex = numRows; 
        end
        
        middleIndex = floor((startIndex + endIndex) / 2); 
        value_in_middle = DataDetails(middleIndex, 17); 
        middleValues(j, i) = value_in_middle;      
        segment = DataDetails(startIndex:endIndex, i);
        rmsValues(j, i) = rms(segment);
    end
end

%% delete unnecessary values
middleValues(:,1:16)=[];

%% add to rms 
rmsValues(:,17)=middleValues;
combinedgestures=rmsValues(:,17);
toreplace=(combinedgestures >= 6 & combinedgestures <= 9);
combinedgestures(toreplace)=4;
rmsValues(:,17)=combinedgestures;

%% plot RMS of electrode 9 level 5
figure;
subplot(2,1,1)
time=(0:length(rmsValues)-1) / fs;
plot(time,rmsValues(:,1));
title('E9 level 5');
xlabel('time');
ylabel('RMS Value');
%xlim([0 0.75]);
subplot(2,1,2)
plot(time,rmsValues(:,17));
title('E9 level 5');
xlabel('time');
ylabel('Gesture Value');
%xlim([0 0.75]);

%% training and testing data 
trainingdata=rmsValues(1:16605,1:16);
testingdata=rmsValues(16606:19013,1:16);
traininggestures=rmsValues(1:16605,17);
testgestures=rmsValues(16606:19013,17);

%% Run SVM model 
tic
svm_model=fitcecoc(trainingdata,traininggestures);
elapsedtime2=toc;
fprintf('Elapsed time for SVM model: %.2f seconds\n', elapsedtime2);

%% create predictions array
predictions=predict(svm_model,testingdata);

%% display accuracy 
accuracy = sum(predictions == testgestures) / numel(testgestures);
disp(['The SVM accuracy is: ', num2str(accuracy*100),'%']);

%% confusion matrix
actual = testgestures;
predicted = predictions;
class_labels = {'Idle', 'Fist', 'flexion', 'Extension', 'Pinch'};
conf_matrix = confusionmat(actual, predicted);
conf_matrix(isnan(conf_matrix)) = 0;
conf_matrix_normalized = conf_matrix ./ sum(conf_matrix, 2);
figure;
imagesc(conf_matrix_normalized);
title('Normalized Confusion Matrix');
colormap(cool);
set(gca, 'XTick', 1:numel(class_labels), 'XTickLabel', class_labels);
set(gca, 'YTick', 1:numel(class_labels), 'YTickLabel', class_labels);
textStrings = num2str(conf_matrix_normalized(:), '%0.2f');
textStrings = strtrim(cellstr(textStrings));
[x, y] = meshgrid(1:numel(class_labels));
hStrings = text(x(:), y(:), textStrings(:), 'HorizontalAlignment', 'center');
xlabel('Predicted label');
ylabel('True label');

%% plot predictions in time domain
timed=(0:length(predictions)-1) / fs;
figure;
plot(timed, predictions);
title('Predicted Hand Gesture');
xlabel('Time (s)');
ylabel('Hand Gesture');
ylim([0 5]);

%% filtering predictions (post-processing)
window_size = 51; 
filtered_data = medfilt1(predictions, window_size);
figure;
subplot(2, 1, 1);
plot(predictions);
title('Original Data');
xlabel('Sample');
ylabel('Value');
ylim([0 5]);
subplot(2, 1, 2);
plot(filtered_data);
title('Filtered Data');
xlabel('Sample');
ylabel('Value');
ylim([0 5]);

%% using threshold to determine hand gesture 
threshold_value = 0.5;
figure;
next_values = [];
plot(filtered_data);
hold on;
threshold_line = refline([0 threshold_value]);
title('Filtered Data with Threshold');
xlabel('Sample');
ylabel('Value');
ylim([0 5]);
hold off;
first_intersection_idx = find((filtered_data(1:end-1) <= threshold_value & filtered_data(2:end) > threshold_value) | ...
    (filtered_data(1:end-1) >= threshold_value & filtered_data(2:end) < threshold_value), 1);
if ~isempty(first_intersection_idx)
    first_y_value = filtered_data(first_intersection_idx - 1);
    next_values = [first_y_value];
end
for i = first_intersection_idx:length(filtered_data)-1
    if (filtered_data(i) <= threshold_value && filtered_data(i+1) > threshold_value) || (filtered_data(i) >= threshold_value && filtered_data(i+1) < threshold_value)
        next_values = [next_values, filtered_data(i+1)];
    end
end

%% confusion matrix
actual = testgestures;
predicted = filtered_data;
class_labels = {'Idle', 'Fist', 'flexion', 'Extension', 'Pinch'};
conf_matrix = confusionmat(actual, predicted);
conf_matrix(isnan(conf_matrix)) = 0;
conf_matrix_normalized = conf_matrix ./ sum(conf_matrix, 2);
figure;
imagesc(conf_matrix_normalized);
title('Normalized Confusion Matrix');
colormap(cool);
set(gca, 'XTick', 1:numel(class_labels), 'XTickLabel', class_labels);
set(gca, 'YTick', 1:numel(class_labels), 'YTickLabel', class_labels);
textStrings = num2str(conf_matrix_normalized(:), '%0.2f');
textStrings = strtrim(cellstr(textStrings));
[x, y] = meshgrid(1:numel(class_labels));
hStrings = text(x(:), y(:), textStrings(:), 'HorizontalAlignment', 'center');
xlabel('Predicted label');
ylabel('True label');

%% Play audio files 
% duration = 300/fs;
% frequencies = [500, 750, 1000, 1500, 2000]; 
% for i = 1:length(filtered_data)
%     value = filtered_data(i);
%     frequency = frequencies(value + 1); 
%     tone = sin(2 * pi * frequency * (0:1/fs:duration));
%     transtone=transpose(tone);
%     sound(transtone, fs);  
%     disp(frequency);
% end

%% piano 
[y1,sf1]=audioread('sound1.wav');
[y2,sf2]=audioread('sound2.wav');
[y3,sf3]=audioread('sound3.wav');
[y4,sf4]=audioread('sound4.wav');
[y5,sf5]=audioread('sound5.wav');
sound(y1,sf1);
gesturessound=next_values;
for i = 1:length(gesturessound)
    switch gesturessound(i)
        case 0
            sound_to_play = y1;
            sample_rate = sf1;
            disp('Gesture 0');
        case 1
            sound_to_play = y2;
            sample_rate = sf2;
             disp('Gesture 1');
        case 2
            sound_to_play = y3;
            sample_rate = sf3;
             disp('Gesture 2');
        case 3
            sound_to_play = y4;
            sample_rate = sf4;
             disp('Gesture 3');
        case 4
            sound_to_play = y5;
            sample_rate = sf5;
             disp('Gesture 4');
        otherwise
            disp('Invalid gesture sound value.');
            continue; 
    end
    
    sound(sound_to_play, sample_rate);
    pause(length(sound_to_play) / sample_rate);
end