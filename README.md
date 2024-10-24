# Music Genre Recognition - A research conducted to examine the effect of song length on the quality of classification into a musical genre

The repository contains source code and data related to a master's thesis that investigated the effect of the length of audio segment lengths on the quality of music genre classification using deep neural networks. The study compared two sound representations: the Mel spectrograms and MFCC coefficients, and the CNN and ResNet50 classification models. Experimental results showed that Mel spectrograms and audio segment lengths of 5-10 seconds provide the best results for GTZAN dataset, especially for the ResNet50 model. The research has applications in music recommendation systems and automatic song tagging.

## Table of contents
- [Code](#code)
- [Data](#data)
- [Project Structure](#project-structure)
- [Models architecture](#models-architecture)
- [Cross-validation](#cross-validation)
- [Results](#results)
- [References](#references)
  
## Code
The project includes several Jupyter notebooks, each addressing a specific aspect of the implementation:
- [**`standard_models_classification.ipynb`**](standard_models_classification.ipynb): Implements a supervised learning approach to train and classify audio tracks using CSV files from both the GTZAN and FMA datasets, also includes feature extraction techniques and  the construction of basic neural networks
- [**`track_segmentation.ipynb`**](track_segmentation.ipynb): Provides tools for track splitting and directory-level operations to efficiently manage the dataset
- [**`nn_models_training.ipynb`**](nn_models_training.ipynb): Covers the creation and training processes for the neural network models
- [**`nn_models_results.ipynb`**](nn_models_results.ipynb): Contains the results and visualizations generated from the neural network models
  
## Data
All data utilized in this project is stored in the [Data](Data/) directory. It includes the original [GTZAN dataset](Data/GTZAN/) as well as custom-generated [spectrograms](Data/Spectrograms), which distinguish between standard, Mel and MFCC spectrograms. Additionally, the following compressed zip files for each data split are available for download:
- [**`spectrograms_1s.zip`**](https://drive.google.com/file/d/1vxjNcUdFdhW1p7wTddkkSll-HxP8K-7W/view?usp=sharing)
- [**`spectrograms_3s.zip`**](https://drive.google.com/file/d/1YxlklfsPRNBh-n-lTQm2Q8A0YWZdnNt9/view?usp=drive_link)
- [**`spectrograms_5s.zip`**](https://drive.google.com/file/d/1OwveDXfZGhnc36ecxEDpujncuTEH2mX7/view?usp=drive_link)
- [**`spectrograms_10s.zip`**](https://drive.google.com/file/d/1RoRnAqqCYD6R5pAATPDwkRC25Jd5VMBN/view?usp=drive_link)
- [**`spectrograms_20s.zip`**](https://drive.google.com/file/d/19ZWyA6fu_w7OoCJemZeBvWtd7PjO8WRu/view?usp=drive_link)
- [**`spectrograms_full_length.zip`**](https://drive.google.com/file/d/1bmEvXJJ8nP5iQArlmckXFacq0nXMwyF3/view?usp=drive_link)

## Project Structure

![Diagram bez tytułu drawio (2)](https://github.com/user-attachments/assets/5303f06b-feff-42fa-b7a5-9dc3b3d4cc9d)

This diagram illustrates the workflow for classifying music genres using deep neural networks:

- **Music sample**: The process begins with a music sample, visualized as a waveform that represents amplitude over time.

- **Spectrogram generation**: The audio waveform is converted into a spectrogram, which displays the intensity of frequencies over time. This step transforms the raw audio into a visual representation that can be used for feature extraction.

- **Model training**:

  - **Feature extraction**: Convolutional Neural Networks (CNNs) are used to extract relevant features from the spectrogram.
  - **Classification**: The extracted features are then passed through fully connected layers for classification, where the model learns patterns characteristic of different music genres.
- **Trained model**: Once the model is trained, it can be used to classify new music samples.

- **Predicted Genre**: The final output of the process is the predicted genre for the input music sample.

## Models architecture
![Diagram bez tytułu (2)](https://github.com/user-attachments/assets/b1a1b497-3a78-4068-be9a-5184be263d50)

## Cross-validation
To ensure the reliability of the results and minimize the impact of random data partitioning, a **custom Monte Carlo cross-validation** method was employed. This approach involved randomly splitting the dataset into training (80%) and testing (20%) subsets on three separate occasions. Unlike traditional k-fold cross-validation, which divides data into equal parts, this method facilitates complete randomness in data partitioning. Each model was trained and evaluated on three distinct, representative subsets, leading to more stable outcomes. The final accuracy for each model was determined by calculating the average of these three values, thereby reducing the influence of randomness and providing a more objective assessment.

## Results
The study successfully demonstrated a significant influence of song length on the quality of classification. Models achieved competitive results, with **Model_1 attaining an average accuracy of 63.99%** and **Model_ResNet50 achieving an average accuracy of 72.84%**. These results are comparable to other studies in the field, affirming the effectiveness of the approach.

![Zrzut ekranu 2024-10-23 221601](https://github.com/user-attachments/assets/575315ed-67fd-458d-b59b-f477bb77529a)

## References
[T. Shaikh, A.Jadhav (2022) Music Genre Classification Using Neural Network](https://www.researchgate.net/publication/360417353_Music_Genre_Classification_Using_Neural_Network)  
[A. ELbir, N. Aydin (2020) Music genre classification and music recommendation by using deep learning](https://www.researchgate.net/publication/339678441_Music_Genre_Classification_and_Music_Recommendation_by_Using_Deep_Learning)  
[Mehta et al. (2021)](https://www.researchgate.net/publication/351379331_Music_Genre_Classification_using_Transfer_Learning_on_log-based_MEL_Spectrogram)  
[Ndou et al. (2021)](https://www.researchgate.net/publication/351595695_Music_Genre_Classification_A_Review_of_Deep-Learning_and_Traditional_Machine-Learning_Approaches)  
[Li et al. (2021)](https://link.springer.com/article/10.1007/s11042-020-10465-9)  
[G. Tzanetakis, P. Cook (2002)](https://www.researchgate.net/publication/3333877_Musical_Genre_Classification_of_Audio_Signals)  

## License
This project is licensed under the MIT License. Feel free to use, modify, and distribute the application as per the terms of the license.






