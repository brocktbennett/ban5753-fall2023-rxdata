# GeoAI Competition: The Hyperspectral Imaging Challenge

**Date Started:** 12 November 2023
**Date Completed:** 08 December 2023

**Team Name:**  
MTTDATA (AKA: RxData)

**Team Members:**  
- Brock Bennett
- Isabella Lieberman
- John Ramirez
- Nathan Zlomke

# GeoAI Competition: The Hyperspectral Imaging Challenge 🛰️🌾

![Date](https://img.shields.io/badge/Date-8%20December%202023-blue)
![Team Name](https://img.shields.io/badge/Team%20Name-Team%2013%3A%20RXDATA-green)
![Team Members](https://img.shields.io/badge/Team%20Members-Brock%20Bennett%20(A20128702)%2C%20Isabella%20Lieberman%20(A20339586)%2C%20John%20Ramirez%20(A20152887)%2C%20Nathan%20Zlomke%20(A10912611)-purple)

## Table of Contents
- [Executive Summary](#Executive-Summary)
- [Introduction](#Introduction)
- [Project Objectives](#Project-Objectives)
- [Data Introduction/Sourcing](#Data-Introductionsourcing)
- [Data Preparation](#Data-Preparation)
- [Data Visualizations](#Data-Visualizations)
- [Data Analysis](#Data-Analysis)
- [Data Generalizations/Explanations](#Data-Generalizationsexplanations)
- [Future Scope](#Future-Scope)
- [Conclusions](#Conclusions)
- [References](#References)
- [Appendix](#Appendix)
  - [Appendices A – Data Visualizations](#appendices-a--data-visualizations)
  - [Appendices B - Project Repository and Source Files](#appendices-b---project-repository-and-source-files)

## Executive Summary 📜

The challenge involves automating the estimation of essential soil parameters—potassium, phosphorus, magnesium, and pH—from hyperspectral data gathered over agricultural regions in Poland. After performing data transformations, the LightGBM model emerged as the most effective solution.

The potential of hyperspectral technology and advanced machine learning to optimize agriculture applications will have cascading benefits in addressing contemporary challenges related to water scarcity, food security, and environmental impacts. Further benefits may be gleaned from further enhancing resolution and accuracy within specific wavelength ranges for a more nuanced understanding of soil parameters, contributing to improved sustainable agriculture practices.

Insights gained from the analysis of key spectral features influencing soil parameter estimations guide future applications. A broad scope into our insights and recommendations is that all soil parameters aside from magnesium are highly significant in making predictions.

## Introduction 🌱

ESA Φ-lab, in collaboration with KP Labs and their partner QZ Solutions, has launched an extraordinary initiative to transform agriculture's future by leveraging in-orbit processing. The primary goal is to enhance farm sustainability by utilizing the latest advancements in Earth observation and artificial intelligence. This approach helps address the challenge of affordable food production and contributes to environmentally friendly agriculture practices.

One of the critical aspects is providing farmers with timely information about soil parameters to optimize their fertilization processes. This optimization can lead to selecting more suitable fertilizer mixes and reducing overall fertilizer usage. Currently, the traditional method for quantifying soil parameters is labor-intensive and time-consuming. It involves collecting soil samples in the field and sending them to specialize labs for chemical analysis. Moreover, the limited number of sampling points in the field, often spread across large areas, needs to be improved to ensure the accuracy of test results. In-situ analysis could be more scalable and more time-inefficient.

The proposed solution is to harness innovative airborne and satellite hyperspectral imaging technology to promote more sustainable agriculture practices, contributing to a better future for our planet.

## Project Objectives 🚀

The AI for Good initiative, led by ITU in collaboration with 40 UN Sister Agencies, is dedicated to harnessing the power of artificial intelligence to advance the United Nations Sustainable Development Goals on a global scale. As the premier action-oriented UN platform on AI, its mission is to identify practical applications that drive positive impact. This challenge specifically focuses on advancing soil parameter retrieval from hyperspectral data to prepare for the upcoming Intuition-1 mission.

## Data Introduction/Sourcing 📊

To train the machine learning model, ground truth samples were collected for each soil parameter. The model will identify which unique features, or bandwidths, have associations with each parameter. The dataset consists of 2886 patches that are, on average, 60 by 60 pixels. Each pixel within the patch contains readings for 150 wavelength bands (462-942 nm). An example image of Band 100 is presented in the Appendix in Figure 1.

The distributions of each soil parameter were plotted and are presented below:

[Insert Data Distribution Visualizations]

## Data Preparation 🧹

During data exploration, it was discovered that the distributions of some soil parameters were skewed. Since these are target variables, it is not recommended to perform power series transformations; however, Yeo-Johnson transformations were performed on predictor variables. These transformations can help achieve better accuracy in predicting the target variables.

Predictor and target variables were scaled before initializing the model, ensuring all variables are in the same scale. This prevents higher magnitudes from biasing the model and lessens the impact of outliers.

## Data Visualizations 📊

*Please see Visualizations and Figures in the Appendix*

## Data Analysis 📈

Chemometric models are used to reveal trends in chemical data; in this case, calibrating spectral data. Models that were considered included Partial Least Squares (PLS) regression, Support Vector Regression (SVR), Convolutional Neural Networks (CNN), and the Light Gradient Boosting Model (LightGBM).

Partial Regression: Initially, PLS proved to be one of the best-performing, simplest models.

Support Vector Regression: SVR was explored as another model. This model attempts to find a hyperplane in the dataset that results in a minimum error between training and predicted values.

Convolutional Neural Networks: CNN is a deep learning framework often used in image classification and works well for hyperspectral matrices.

Light Gradient Boosting: The last chemometric model to be evaluated was the LightGBM regression model.

Distributions of training soil parameter values compared to predicted values were plotted.

[Insert Model Comparison Visualizations]

## Data Generalizations/Explanations 📚

### Feature Importance
To assess feature performance, SHAP (SHapley Additive exPlanations) was employed to better understand what features most impact each soil parameter and to what degree.

#### Phosphorus
[Insert Phosphorus Feature Importance Visualizations]

#### Potassium
[Insert Potassium Feature Importance Visualizations]

#### Magnesium
[Insert Magnesium Feature Importance Visualizations]

#### Soil pH
[Insert Soil pH Feature Importance Visualizations]

### Insights & Recommendations

#### Phosphorous
- Precision in administering phosphorous is crucial for crops and water bodies.
- Specialize phosphorous reading between Bandwidth 39 to 48 with higher resolution for better predictions.

#### Potassium
- Research analyzing potassium using hyperspectral models is pivotal for water conservation.
- Enhance precision in the spectral bandwidth range of 50 to 70.

#### Magnesium
- Wide-spectrum measurements are essential due to water scarcity.

#### Soil pH
- Enhance predictions for soil pH by focusing on bandwidths 39 to 72.

### Future Scope 🚀

Hyperspectral technology paired with powerful machine learning techniques is pivotal in tackling current global challenges like water scarcity, food security, and environmental protection. Future advancements and applications could include:

- Improved Spectral Resolution: Narrowing bandwidths for specific soil parameters can significantly refine the specialized characterization of soil parameters, thereby improving prediction accuracy.

## Conclusions 🌍

In collaboration with ESA Φ-lab, KP Labs, and QZ Solutions, our study focused on optimizing the LightGBM model for automated soil parameter estimation, representing a significant stride in precision agriculture. Key findings emphasize the importance of fine-tuning phosphorus predictions within the 39 to 48 bandwidth range, highlighting the need for calibration and resolution enhancements.

Moreover, the study identifies critical spectral bandwidths in the 50 to 70 range for potassium predictions, showcasing the transformative impact of hyperspectral technology in advancing sustainable farming practices. The research underscores the potential of machine learning, particularly the LightGBM model, in contributing to the evolution of agricultural technology.

As we continue to refine models and explore innovative applications, these insights mark a milestone in our collaborative efforts to harness advanced technologies for the betterment of global agriculture. The study's findings provide actionable recommendations for future research and application, laying the groundwork for a more data-driven and efficient approach to soil parameter estimation in the agricultural sector.

## References 📖

- [Fresh Water Resources in Poland](https://www.climatechangepost.com/poland/fresh-water-resources/)
- [Global Hyperspectral Imaging Market Report](https://www.globenewswire.com/en/news-release/2022/04/01/2414594/28124/en/Global-Hyperspectral-Imaging-Market-Report-2022-2030-Rising-Need-for-In-depth-Data-from-the-Optical-Images-will-Increase-Demand-for-Hyperspectral-Imaging-Systems.html)
- [High-magnesium waters and soils: Emerging environmental and food security constraints](https://pubmed.ncbi.nlm.nih.gov/30045492/#:~:text=A%20ratio%20of%20magnesium%2Dto,and%20impact%20crop%20yields%20negatively)
- [Farmers Are Facing a Phosphorus Crisis. the Solution Starts with Soil](https://www.nationalgeographic.com/science/article/farmers-are-facing-a-phosphorus-crisis-the-solution-starts-with-soil)
- [What Is Hyperspectral Imaging?](https://www.specim.com/technology/what-is-hyperspectral-imaging/)

## Appendix 📄

### Appendices A – Data Visualizations

[Insert Data Visualization Figures]

### Appendices B - Project Repository and Source Files

[GitHub Link](https://github.com/osu-msba/ban5753-fall2023-rxdata/tree/main/GeoAI-Competition-23/starting_packCNN)
[Source Files](https://drive.google.com/drive/folders/1o3rHMzK-2HBP8ZfS-9GeAW_kqXAycEsP?usp=share_link)

