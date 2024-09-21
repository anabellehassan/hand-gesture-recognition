# EMG-Controlled Musical Tone Generator

This project implements a MATLAB-based framework designed to generate musical tones based on EMG (Electromyography) signals obtained from voluntary hand gestures. Machine Learning is applied to recognize gestures from sEMG data. Based on the recognized gestures, the system adjusts the pitch (frequency) of the generated sound, effectively allowing users to control music through hand movements.

## Features

•⁠  ⁠*EMG Signal Processing*: The system reads and processes raw EMG signals from CSV files, specifically focusing on select electrodes.

•⁠  ⁠*Data Preprocessing*: Unnecessary data, such as relaxed state signals, are removed from the dataset, focusing only on relevant gestures.

•⁠  ⁠*Gesture Recognition*: Hand gestures are recognized using machine learning techniques, with data filtered and cleaned to enhance recognition accuracy.

•⁠  ⁠*Real-Time Visualization*: The system generates plots of the raw and processed EMG data, allowing for easy visualization of gestures and signal strength.

•⁠  ⁠*Sound Generation*: The recognized hand gestures are mapped to musical tones.

## Code Workflow

1.⁠ ⁠*Data Loading*: The EMG data is loaded from a CSV file, which contains the raw signals from multiple electrodes.

2.⁠ ⁠*Raw Data Visualization*: Plots of the raw data are generated to visualize the EMG signals from specific electrodes before any processing.

3.⁠ ⁠*Data Cleaning*: Signals from unnecessary electrode bands are removed, and the relaxed state gestures (labeled ⁠ -1 ⁠) are filtered out to focus on relevant hand gestures.

4.⁠ ⁠*Processed Data Visualization*: After data cleaning, the system plots the processed EMG data, highlighting the gestures over time.

5.⁠ ⁠*Gesture Transition Detection*: Changes in hand gestures are detected by identifying transitions between different signal values.

6.⁠ ⁠*Sound Generation*: Based on the detected gestures, corresponding musical tones are generated with varying pitches.

## Requirements

•⁠  ⁠MATLAB (version 2020b or later recommended)

•⁠  ⁠Signal Processing Toolbox

•⁠  ⁠Machine Learning Toolbox

•⁠  ⁠Access to EMG data in CSV format (provided separately)

## Installation

1.⁠ ⁠Clone this repository to your local machine:
    ⁠ bash
    git clone https://github.com/yourusername/emg-musical-tone-generator.git
     ⁠
   
2.⁠ ⁠Install the required MATLAB toolboxes:
    - Signal Processing Toolbox
    - Machine Learning Toolbox

3.⁠ ⁠*Download the source EMG data files* from the link below, as they are too large to include directly in this repository:

[Download HandGestureData.csv](https://drive.google.com/file/d/1RUpmCO2BttxLqYAg5LB1GGCRE0m3KAvo/view?usp=share_link)

[Download DataDetails.mat](https://drive.google.com/file/d/1wqZZR037j8IpGq2ZLHm2YV6djsi1vR5A/view)

4.⁠ ⁠Place the downloaded data files in the appropriate directory specified in the MATLAB code.

## Usage

1.⁠ ⁠Load the EMG data into the system by running the script by sections from ⁠ HandGestureFinal.m ⁠ in MATLAB.

2.⁠ ⁠The system will process the data, plot relevant visualizations, and recognize hand gestures.

3.⁠ ⁠The system generates real-time musical tones based on the recognized gestures, adjusting the pitch according to your movements.

## Performance

The system has demonstrated an average accuracy of over 90% in hand gesture recognition.

## License

This project is licensed under the MIT License.
