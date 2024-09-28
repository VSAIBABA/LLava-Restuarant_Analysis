# LLava-Restuarant_Analysis
This repository provides a solution for restaurant image analysis using the LLava model. The goal is to automate the process of analyzing restaurant images by identifying elements such as food dishes, people, and other objects. The system also evaluates the image quality and provides insights on restaurant operations based on visual data.

Table of Contents
Introduction
Features
Installation
Usage
Running the Analysis
Input Data Format
Output Results
Contributing
License
Introduction
The LLava Restaurant Analysis project leverages the LLava model to perform restaurant image analysis tasks such as:

Describing food dishes.
Counting the number of dishes or people in the images.
Evaluating the quality of the image on a scale from amateur to professional level.
The model reads images and corresponding instructions from a dataset and automates the analysis process by generating outputs based on user-defined prompts.

Features
Automated Image Analysis: Processes restaurant images to identify elements like food dishes, people, and tables.
Quality Evaluation: Ranks image quality on a predefined scale.
Object Counting: Counts the number of visible dishes or people in the image.
Customizable Instructions: Instructions can be provided via a CSV file to define the type of analysis required for each image.
Installation
To get started with this project, follow the steps below:

Prerequisites
Python 3.8 or higher
Git
A GPU-enabled environment with CUDA support is recommended.
Clone the Repository
bash
Copy code
git clone -b sample https://github.com/VSAIBABA/LLava-Restuarant_Analysis.git
cd LLava-Restuarant_Analysis
Install Dependencies
bash
Copy code
pip install -r requirements.txt
Additional Setup
Install the LLava model with specific transformers and torch versions:
bash
Copy code
pip install torch transformers==4.36.2
Download the LLava model for restaurant analysis as described in the project files.
Usage
Running the Analysis
To run the analysis on a dataset of images, you can modify the script to point to your image directory and instruction file. Then run the analysis with the following command:

bash
Copy code
python analyze_images.py --image_dir /path/to/images --instructions_file /path/to/instructions.csv
Input Data Format
Image Directory: All images should be stored in a single directory.

Instructions File: The instructions file should be in CSV format, with each row containing an instruction for an image. Example format:

csv
Copy code
instructions
"Describe the image and color details."
"Count the number of food dishes."
"Evaluate the picture quality on a scale of 1 to 3."
Output Results
After running the analysis, the output will be printed directly to the console, showing the results of the analysis for each image and instruction. The output includes:

Descriptions of the image.
The number of identified objects (e.g., dishes, people).
The quality score of the image.
Contributing
We welcome contributions to this project! If you'd like to contribute, please fork the repository and submit a pull request.

Fork the repository.
Create a new branch (git checkout -b feature-branch).
Commit your changes (git commit -m "Add new feature").
Push to the branch (git push origin feature-branch).
Open a pull request.
License
This project is licensed under the MIT License - see the LICENSE file for details.


