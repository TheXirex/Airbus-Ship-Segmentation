# Airbus-Ship-Segmentation
## Main Goal
The main goal of this project is to build a semantic segmentation model for ship detection in satellite images.

Also this project is a test assignment for getting an internship in the R&D center WINSTARS.AI.
## Technologies
The created model for ship segmentation has Unet architecture (the code of which you can find in utils/unet.py file).

To train the model I used Focalloss loss function and dice_score binary similarity method (code for the methods can be found in utils/losses.py).

## EDA
Analysis of the dataset and model metrics can be found behind the files: [eda.ipynb](eda.ipynb) and [model_metrics](model_metrics)
## Installation
1. The Python version for this project is **3.11.5**.
2. Select the directory where the project is to be loaded.
3. Go to this directory in the console and clone the repository:
```
git clone https://github.com/TheXirex/Airbus-Ship-Segmentation.git
```
4. Browse to the repository folder:
```
cd Airbus-Ship-Segmentation
```
5. Install the required libraries:
```
pip install -r requirements.txt
```
6. There are 2 ways to demonstrate how the model works:
  - web application built on streamlit.
    ```
    streamlit run inference.py
    ```
  - demonstration of results in .ipynb notepad [test.ipynb](test.ipynb).
7. If you want to retrain a model with your parameters:
  - Create a 'data' folder in the root of the project.
  - Download the image archive from [Kaggle](https://www.kaggle.com/competitions/airbus-ship-detection/data) and upload its contents to the 'data' folder.
  - Change the required parmeters in the [config file](config.py) and run the [training file](train.py).
## Results:
A trained model for semantic segmentation of ships in satellite images.

The model does a good job of segmenting explicit ships in images, but sometimes gets confused with shore/land areas in images.

Example images:
![photo_2023-11-20_15-02-38](https://github.com/TheXirex/Airbus-Ship-Segmentation/assets/104722568/8f80bd2a-c5bf-4c6a-a760-e30ad4587c6a)
![изображение](https://github.com/TheXirex/Airbus-Ship-Segmentation/assets/104722568/f119072a-8d0b-444f-9560-610d3bdcb217)
![изображение](https://github.com/TheXirex/Airbus-Ship-Segmentation/assets/104722568/a15e4377-9d11-4671-92a3-58de5dddd929)


