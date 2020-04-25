# EMG-based Gesture Recognition using Capsule Network
### Data
The NinaPro dataset will be downloaded automatically by running the main.py script if it is not found.

### Requirements
1. python 3.7.4
2. numpy 1.18.1
3. scikit-image 0.16.2
4. torch 1.4.0
5. tqdm 4.44.1
6. matplotlib 3.1.3

### TODO
- [x] Build dataset downloader and loader
- [x] Build feature extractors
- [x] Build general classification engine
- [x] Build general training/evaluation engine
- [x] Build conventional classifiers (Support Vector Machine, Random Forest)
- [x] Build Neural Network based classifiers (Fully Connected Network, Convolutional Neural Network)
- [x] Build Capsule Network based classifier
- [x] Create logging and plotting engines
- [x] Train and Evaluate

### Train
Multiple data types, feature types and models could be chosen. For instance, the following command line runs a training of a capsule net using the base dataset and the root mean square features.
````
python train.py --model capsnet --features rms --data intrasubjects --epochs 50 --chkpt_period 1 --valid_period 1 --batch_size 5 --lr 0.001 --verbose 1
````
Run `python train.py -h` to see the possible values of the arguments.

### Results
#### Testing on Intrasubject Fixed Window Dataset
| Features |   rf    | SVM  | FcNET  | ConvNET  | CapsNET  |
| -------- |:-------:| ----:| ------:| --------:| --------:|
| rms      | 66.56% | 52.06% | 67.67% | 55.54% | **76.04%** |
| hist      | 56.34% | 38.58% | 49.28% | 53.88% | **65.67%** |
| multirms      |  65.29% | 50.93% | 57.26% | 61.47% | **77.85%** |
| pmrms      |  66.44% | 63.15% | 63.29% | 61.72% | **76.41%** |
| kmrms      | 54.4% | 25.57% | 38.73% | 39.65% | **67.47%** |
| fourier      | 62.32% | 48.34% | 57.14% | 56.77% | **69.18%** |


#### Testing on Intrasubject Variable Window Dataset
| Features |   rf    | SVM  | FcNET  | ConvNET  | CapsNET  |
| -------- |:-------:| ----:| ------:| --------:| --------:|
| rms      | 79.53%   | 53.27%| 74.67% |  66.26%    |   **90.56%**  |
| multirms      | 50.93%   | 66.82%| 48.69% |  81.21%    |   **93.27%**  |
| pmrms      | 81.96%   | 82.42%| 48.69% |  81.21%    |   **92.52%**  |
| kmrms      | 64.67%   | 23.92%| 15.88% |  39.65%    |   **85.04%**  |
#### Testing on Intersubject Fixed Window Dataset
| Features |   rf    | SVM  | FcNET  | ConvNET  | CapsNET  |
| -------- |:-------:| ----:| ------:| --------:| --------:|
| rms      | 15%   | 14.8%| **17.42%** |  16.2%    |   15.56%  |
| hist      | 14.82%   | 14.95% | 14.67% |  14.88%    |   **15.21%**  |
| multirms      | 15.97%   | 15.03%| 15.51% |  15.49%    |  **17.57%**  |
| pmrms      | 9.8%   | **13.4%**| 11.93% |  10.97%  |   11.78% |
| kmrms      | 11.85%   | 11.07%| 14.09% |  13.4%  |  **15.00%**  |
| fourier      | 16.02% | 14.95% | 16.98% |  17.42%  | **20.54%** |

#### Testing on Intersubject Variable Window Dataset
| Features |   rf    | SVM  | FcNET  | ConvNET  | CapsNET  |
| -------- |:-------:| ----:| ------:| --------:| --------:|
| rms      | 15.2%  | 15.67%| 19.27% | 19.59%  | **20.84%**  |
| multirms      | 22.88%  | 20.53% | 27.27% |  27.11%  |  **28.99%**  |
| pmrms      | 18.18% | 23.19% | 20.68% |  20.06%    |   **23.51%**  |
| kmrms      | 14.89%   | 10.81% | 12.06% |  18.02%  |   **19.74%**  |