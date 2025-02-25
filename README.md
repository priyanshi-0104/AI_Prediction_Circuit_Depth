# AI_Prediction_Circuit_Depth

AI Algorithm for Estimating Combinational Logic Depth

Overview

This project utilizes machine learning to estimate the combinational logic depth of signals in RTL (Register Transfer Level) designs. Based on RTL design data, the model assists in optimizing circuit performance, minimizing timing violations, and enhancing hardware efficiency.

Project Objectives

Create a predictive model to estimate the depth of combinational logic in RTL circuits.

Apply machine learning methods for feature extraction and training.

Enhance design accuracy and efficiency by detecting critical timing paths.

Setup Instructions

1. Clone the Repository

To begin, clone the project repository with the following command:
git clone <repository_link>
cd AI_Circuit_Depth_Prediction

2. Install Dependencies

Make sure you have Python 3.x installed, then install the dependencies using:

pip install -r requirements.txt

3. Prepare the Dataset

Edit rtl_dataset.csv to add your RTL design data.

Make sure the dataset has necessary features such as gate count, logic type, and fan-out.


4. Train and Test the Model

Execute the script to train and test the machine learning model:

python train_model.py

The script will:

Load the dataset.

Preprocess the data (missing value handling, feature normalization).

Train a machine learning model (Random Forest, XGBoost, etc.).

Test the model using performance metrics such as RMSE and R² score.

Usage Instructions

Place your RTL dataset in rtl_dataset.csv.

Execute train_model.py to make predictions.

Use predict.py to test the model on unseen data.

Dependencies

To execute this project, you should have the following dependencies installed:

Python 3.x – For executing scripts.

Scikit-learn – For training machine learning models.

Pandas – For data preprocessing and handling.

NumPy – For numerical computation.

Matplotlib/Seaborn – (Optional) For plotting feature importance and model accuracy.

Future Enhancements

Integrate deep learning models to enhance prediction accuracy.

Create a web interface for real-time predictions.

Expand the model to handle other RTL design parameters.

Conclusion

This artificial intelligence-based technique gives a strong and effective way of estimating combinational logic depth in RTL designs. Through automation, this reduces the designer's efforts to optimize circuit performance, improve timing analysis, and decrease design iterations.
