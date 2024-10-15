import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import pickle
import json

# Function to load pickle files
def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

# Function to display model results
def display_model_results(results):
    if isinstance(results, dict):
        st.write(f"### Accuracy: {results.get('accuracy', 'N/A'):.4f}")
        st.write("### Classification Report")
        st.text(results.get('classification_report', 'N/A'))
        st.write("### Confusion Matrix")
        sns.heatmap(results.get('conf_matrix_df', pd.DataFrame()), annot=True, fmt='d', cmap='BuPu')
        plt.xlabel('Predicted Class')
        plt.ylabel('Real Class')
        st.pyplot(plt.gcf())
    else:
        st.write("Invalid results format.")

# Set the title of the app
st.markdown("<h1 style='color: purple;'>COVID-19 Radiography Analysis</h1>", unsafe_allow_html=True)

# Set the title of the sidebar
st.sidebar.markdown("<h1 style='color: purple;'>Table of Contents</h1>", unsafe_allow_html=True)

# Define the sidebar navigation using selectbox
sections = {
    "Introduction": {
        "Overview": "",
        "Objectives": "",
        "Perspectives": ""
    },
    "Data Exploration": {
        "Metadata": "",
        "X-Rays and Masks": ""
    },
    "Preprocessing": {
        "Data preparation": "",
        "Region of Interests": "",
        "Filters": "",
        "Data Augmentation": ""
    },
    "Modelling": {
        "Classical Modelling": "",
        "Deep Learning Modelling": ""
    },

    "Interpretability": {
        "Saliency Maps": ""
    },
    "Predictions": {
        "Model predictions": ""
    },
    "Conclusions": {
        "Conclusion": ""
    }
}

# Sidebar for main sections
section = st.sidebar.selectbox("Go to:", list(sections.keys()))

selected_subtitle = None

if section == "Modelling":
    mod_subsection = st.sidebar.radio("Modelling Subsections", ["Classical Modelling", "Deep Learning Modelling"])
    st.write(f"### {mod_subsection}")
else:
    # Display subtitles and content based on the selected section
    st.sidebar.markdown(f"<h2 style='color: gray;'>{section}</h2>", unsafe_allow_html=True)
    if section in sections:
        selected_subtitle = st.sidebar.radio("Subsections", list(sections[section].keys()))
        st.write(f"### {selected_subtitle}")
    else:
        st.write("No subsections available for the selected section.")


# Check for the selected subtitle and display the corresponding content
if section == "Modelling":
    if mod_subsection == "Classical Modelling":
        dataset_choice = st.selectbox("Choose the dataset:", ["Whole X-Rays", "ROI", "ROI with Filters"])
        if dataset_choice == "Whole X-Rays":
            image4_path ='/Users/aleksandrastojanovic/Desktop/data_science_projects/image4.png'
            if os.path.exists(image4_path):
                image4 = Image.open(image4_path)
                st.image(image4, use_column_width=True)
            image5_path = '/Users/aleksandrastojanovic/Desktop/data_science_projects/Image5.png'
            if os.path.exists(image5_path):
                image5 = Image.open(image5_path)
                st.image(image5, use_column_width=True)
            image8_path ='/Users/aleksandrastojanovic/Desktop/data_science_projects/image8.png'
            if os.path.exists(image8_path):
                image8 = Image.open(image8_path)
                st.image(image8, use_column_width=True)
            model_choice = st.selectbox("Choose a model:", ["Random Forest", "Bagging with Random Forest", "XGBoost"])
            if model_choice == "Random Forest":
                results_rf = load_pickle('rf_results.pkl')
                display_model_results(results_rf)   
            elif model_choice == "Bagging with Random Forest":
                results_bagging = load_pickle('bagging_results.pkl')
                display_model_results(results_bagging)
            elif model_choice == "XGBoost":
                results_xgboost = load_pickle('xgboost_results.pkl')
                display_model_results(results_xgboost)

        elif dataset_choice == "ROI":
            image6_path = '/Users/aleksandrastojanovic/Desktop/data_science_projects/image6.png'
            if os.path.exists(image6_path):
                image6 = Image.open(image6_path)
                st.image(image6, use_column_width=True)
            model_choice = st.selectbox("Choose a model:", ["Random Forest", "Bagging with Random Forest", "XGBoost"])
            if model_choice == "Random Forest":
                results_rf_roi = load_pickle('rf_roi_results.pkl')
                display_model_results(results_rf_roi)
            elif model_choice == "Bagging with Random Forest":
                results_bagging_roi = load_pickle('bagging_roi_results.pkl')
                display_model_results(results_bagging_roi)
            elif model_choice == "XGBoost":
                results_xgb_roi = load_pickle('xgb_roi_results.pkl')
                display_model_results(results_xgb_roi)
    
        elif dataset_choice == "ROI with Filters":
            image7_path = '/Users/aleksandrastojanovic/Desktop/data_science_projects/image7.png'
            if os.path.exists(image7_path):
                image7 = Image.open(image7_path)
                st.image(image7, use_column_width=True)

            model_choice = st.selectbox("Choose a model:", ["Random Forest", "Bagging with Random Forest", "XGBoost"])

            if model_choice == "Random Forest":
                results_rf_roi_filtered = load_pickle('rf_roi_filtered_results.pkl')
                display_model_results(results_rf_roi_filtered)

            elif model_choice == "Bagging with Random Forest":
                results_bagging_roi_filtered = load_pickle('bagging_roi_filtered_results.pkl')
                display_model_results(results_bagging_roi_filtered)

            elif model_choice == "XGBoost":
                results_xgboost_roi_filtered = load_pickle('xgb_roi_filtered_results.pkl')
                display_model_results(results_xgboost_roi_filtered)
    
    elif mod_subsection == "Deep Learning Modelling":
        if selected_subtitle in sections["Modelling"]["Deep Learning Modelling"]:
            st.write(f"### {selected_subtitle}")
            st.write(sections["Modelling"]["Deep Learning Modelling"][selected_subtitle])
        else:
            st.write(f"Subtitle '{selected_subtitle}' not found in Deep Learning Modelling.")


# Authors section
st.sidebar.markdown("**Authors:**", unsafe_allow_html=True)
authors = ["Preetha BALAKRISHNAN", "Philipp TRINH", "Paul POURMOUSSAVI", "Aleksandra RANCIC"]
for author in authors:
    st.sidebar.write(author)

# Load and display an image in the sidebar
image_path = '/Users/aleksandrastojanovic/Desktop/data_science_projects/data_scientest.png'
if os.path.exists(image_path):
    image = Image.open(image_path)
    st.sidebar.image(image, use_column_width=True)
else:
    st.sidebar.write("Image file not found.")

# Define the content for each page
if section == "Introduction":
    if selected_subtitle == "Overview":
        #st.markdown("""
            #<div style='text-align: justify;'>
               #COVID-19, caused by the SARS-CoV-2 virus, emerged in Wuhan, China, in late 2019 and quickly became a global pandemic. 
                #The virus spreads through respiratory droplets. Symptoms range from mild (fever, cough) to severe (pneumonia). 
                #Detection methods include RT-PCR, antigen tests, serological tests, genome sequencing, CT scans, and X-rays. 
                #X-rays are crucial for rapid patient assessment in hospitals due to their speed and availability.
                #AI and Machine Learning are transforming medical diagnostics, especially for COVID-19. 
                #AI can automate lung X-ray analysis, providing quick, accurate diagnoses and reducing radiologists' workload. 
            #</div>
        #""", unsafe_allow_html=True)
        image2_path = '/Users/aleksandrastojanovic/Desktop/data_science_projects/1.png'
        if os.path.exists(image2_path):
            image2 = Image.open(image2_path)
            st.image(image2, use_column_width=True)
        # Add images
        data_path = '/Users/aleksandrastojanovic/Desktop/data_science_projects/COVID-19_Radiography_Dataset'
        categories = ['Normal', 'Lung_Opacity', 'Viral Pneumonia', 'COVID']
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        for i, category in enumerate(categories):
            image_dir = os.path.join(data_path, category, 'images')
            sample_image = os.listdir(image_dir)[66]  # Use the first image for demonstration
            with Image.open(os.path.join(image_dir, sample_image)) as img:
                axes[i].imshow(img, cmap='gray')
                axes[i].set_title(category)
                axes[i].axis('off')

        st.pyplot(fig)
        st.markdown(
            """
            <div style='text-align: center; font-size: 14px; font-style: italic;'>
                Examples of X-rays from the Dataset
            </div>
            """,
            unsafe_allow_html=True
        )
        
    elif selected_subtitle == "Objectives":
        #st.markdown("""
            #<div style='text-align: justify;'>
                #The goal is to develop a deep-learning model to detect COVID-19 using lung X-ray images. 
                #This tool will enable faster and more accurate diagnosis, aiding medical experts in early treatment. 
                #Early diagnosis reduces symptom severity and the healthcare system's burden, resulting in decreased hospitalizations and resource usage.
            #</div>
        #""", unsafe_allow_html=True)
        image3_path = '/Users/aleksandrastojanovic/Desktop/data_science_projects/Objectives_image.png'
        if os.path.exists(image3_path):
            image3 = Image.open(image3_path)
            st.image(image3, use_column_width=True)

    elif selected_subtitle == "Perspectives":
        st.write("")

elif section == "Data Exploration":
    if selected_subtitle == "Metadata":
        normal_df = pd.read_excel('../../data/raw/Normal.metadata.xlsx')
        lung_opacity_df = pd.read_excel('../../data/raw/Lung_Opacity.metadata.xlsx')
        viral_pneumonia_df = pd.read_excel('../../data/raw/Viral Pneumonia.metadata.xlsx')
        covid_df = pd.read_excel('../../data/raw/COVID.metadata.xlsx')

        st.markdown('''
            * Kaggle: https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database/data
                ''')
        image0 = Image.open('images/dataset.png')
        st.image(image0)

        st.write('### Total count of images categorized by classes')
        st.markdown('''
                * Total of 21,165 images in the dataset
                ''')   
        # count of images from meta data
        count_of_images = [normal_df['FILE NAME'].count(), lung_opacity_df['FILE NAME'].count(), viral_pneumonia_df['FILE NAME'].count(), covid_df['FILE NAME'].count()]
        count_df = pd.DataFrame({'Count of Images': count_of_images}, index = ['Normal', 'Lung Opacity', 'Pneumonia', 'COVID'])
        #st.dataframe(count_df)

        fig = plt.figure(figsize=(8, 4))
        bars = plt.bar(x = range(1,5), height = count_of_images, width = 0.5)
        for bar in bars:
            yval = round(bar.get_height(), 2)
            plt.text(bar.get_x() + 0.125, yval + .005, yval)
        plt.ylabel('Number of images')
        plt.xticks([1, 2, 3, 4], ['Normal', 'Lung Opacity', 'Pneumonia', 'COVID']);
        st.pyplot(fig)

        st.markdown('''
            * metadata.xlsx:
             ''')

        choice_of_class_meta = ['Normal', 'Lung Opacity', 'Viral Pneumonia', 'COVID']
        option_class_meta = st.selectbox('Choice of class', choice_of_class_meta)
        if option_class_meta == 'Normal':
            st.dataframe(normal_df.sample(5).set_index('FILE NAME'))
            st.write('''
                Count of images taken from respective sources:
                ''')
            st.dataframe(pd.DataFrame(normal_df.URL.value_counts()))
        if option_class_meta == 'Lung Opacity':
            st.dataframe(lung_opacity_df.sample(5).set_index('FILE NAME'))
            st.write('''
                Count of images taken from respective sources:
                ''')
            st.dataframe(pd.DataFrame(lung_opacity_df.URL.value_counts()))
        if option_class_meta == 'Viral Pneumonia':
            st.dataframe(viral_pneumonia_df.sample(5).set_index('FILE NAME'))
            st.write('''
                Count of images taken from respective sources:
                ''')
            st.dataframe(pd.DataFrame(viral_pneumonia_df.URL.value_counts()))
        if option_class_meta == 'COVID':
            st.dataframe(covid_df.sample(5).set_index('FILE NAME'))
            st.write('''
                Count of images taken from respective sources:
                ''')
            st.dataframe(pd.DataFrame(covid_df.URL.value_counts()))

    elif selected_subtitle == "X-Rays and Masks":
            st.markdown('''
                * Choose and display samples for comparison:
                ''')         

            def get_images(category, number: int):
                if type(number) == int and number <= len(os.listdir(f'../../data/preprocessed/raw/{category}')):
                    img = Image.open(f'../../data/preprocessed/raw/{category}/{category}-{number}.png')
                    img = img.convert('RGBA')
                    mask = Image.open(f'../../data/raw/{category}/masks/{category}-{number}.png')
                    mask_alpha = mask
                    mask_alpha = mask_alpha.convert('RGBA')
                    mask_alpha.putalpha(90)
                    overlay = Image.alpha_composite(img, mask_alpha)
                else:
                    print('No correct input yet or the chosen category does not have an image with this number.')
                
                return img, mask, overlay

            choice_of_class_images = ['Normal', 'Lung_Opacity', 'Viral Pneumonia', 'COVID']
            option_class_images1 = st.selectbox('First Image: Choose a class', choice_of_class_images)
            number_of_image1 = st.number_input('First Image: Insert a number', min_value=1, value=None, placeholder="Type a number...")
            
            if type(number_of_image1) == int and number_of_image1 <= len(os.listdir(f'../../data/preprocessed/raw/{option_class_images1}')):
                img1, mask1, overlay1 = get_images(option_class_images1, number_of_image1)

                fig_images1 = plt.figure(figsize=(10,10))
                plt.subplot(131)
                plt.imshow(img1)
                plt.title(f'{option_class_images1} - Image {number_of_image1}')
                plt.xticks([])
                plt.yticks([])

                plt.subplot(132)
                plt.imshow(mask1)
                plt.title(f'{option_class_images1} - Mask {number_of_image1}')
                plt.xticks([])
                plt.yticks([])

                plt.subplot(133)
                plt.imshow(overlay1)
                plt.title(f'{option_class_images1} - Overlay')
                plt.xticks([])
                plt.yticks([])

                st.pyplot(fig_images1)

            else:
                st.write('No correct input yet or the chosen category does not have an image with this number.')

            option_class_images2 = st.selectbox('Second Image: Choose a class', choice_of_class_images)
            number_of_image2 = st.number_input('Second Image: Insert a number', min_value=1, value=None, placeholder="Type a number...")

            if type(number_of_image2) == int and number_of_image2 <= len(os.listdir(f'../../data/preprocessed/raw/{option_class_images2}')):
                img2, mask2, overlay2 = get_images(option_class_images2, number_of_image2)

                fig_images2 = plt.figure(figsize=(10,10))
                plt.subplot(131)
                plt.imshow(img2)
                plt.title(f'{option_class_images2} - Image {number_of_image2}')
                plt.xticks([])
                plt.yticks([])

                plt.subplot(132)
                plt.imshow(mask2)
                plt.title(f'{option_class_images2} - Mask {number_of_image2}')
                plt.xticks([])
                plt.yticks([])

                plt.subplot(133)
                plt.imshow(overlay2)
                plt.title(f'{option_class_images2} - Overlay')
                plt.xticks([])
                plt.yticks([])

                st.pyplot(fig_images2)

            else:
                st.write('No correct input yet or the chosen category does not have an image with this number.')

            st.write('#### Images: Tables of Pixels')
            st.markdown('''
                        * grayscale: pixel value between 0 and 255
                        * 25 pixels image and respective table:
                        ''')
            
            image1 = Image.open('images/25pixel_grayscale_img.png')
            image2 = Image.open('images/table_grayscale_img.png')
            images = [image1, image2]

            col1, col2 = st.columns(2)
            with col1: 
                st.image(image1, width=200)
            with col2:
                st.image(image2, width=200)

            st.write('''
                    * colored: red, green, blue value combined
                    * 25 pixel image in RGB space and respective table:
                    ''')
            
            image3 = Image.open('images/25pixel_color_img.png')
            image4 = Image.open('images/table_color_img.png')
            images = [image3, image4]

            col1, col2 = st.columns(2)
            with col1: 
                st.image(image3, width=200)
            with col2:
                st.image(image4, width=400)

            st.write('#### X-Ray Pixel Statistics')
            st.write('''
                    * averaged means, medians and standard deviations of pixel values
                    * all images, masks and ROIs (region of interest):
                    ''')

            image5 = Image.open('images/array_stats.png')
            st.image(image5)

            st.markdown('''
                        * tendency: averaged mean and median pixel values of the Covid images and ROIs are higher
                        ''')
            
            st.write('#### Pixel Intensities for Image, Mask and ROI')
            st.markdown('''
                        * Choose and display samples of pixel intensities for comparison:
                        ''')         

            def get_pixel_intensities(category, number: int):
                if type(number) == int and number <= len(os.listdir(f'../../data/preprocessed/raw/{category}')):
                    img = Image.open(f'../../data/preprocessed/raw/{category}/{category}-{number}.png')
                    pixel_values_img = np.array(img).flatten()
                    mask = Image.open(f'../../data/raw/{category}/masks/{category}-{number}.png')
                    pixel_values_mask = np.array(mask).flatten()
                    roi = Image.open(f'../../data/preprocessed/roi/{category}/{category}-{number}.png')
                    pixel_values_roi = np.array(roi).flatten()
                else:
                    print('No correct input yet or the chosen category does not have an image with this number.')
                
                return pixel_values_img, pixel_values_mask, pixel_values_roi

            option_class_images3 = st.selectbox('Pixel Intensities - First Image: Choose a class', choice_of_class_images)
            number_of_image3 = st.number_input('Pixel Intensities - First Image: Insert a number', min_value=1, value=None, placeholder="Type a number...")
            
            if type(number_of_image3) == int and number_of_image3 <= len(os.listdir(f'../../data/preprocessed/raw/{option_class_images3}')):
                p_v_img1, p_v_mask1, p_v_roi1 = get_pixel_intensities(option_class_images3, number_of_image3)

                fig_images3 = plt.figure(figsize=(16, 6))
                plt.subplot(131)
                plt.hist(p_v_img1, bins = 50, color = 'purple', alpha = 0.7)
                plt.ylim([0,13000])
                plt.title(f'{option_class_images3} - Image {number_of_image3} Pixel Intensity')

                plt.subplot(132)
                plt.hist(p_v_mask1, bins = 50, color = 'purple', alpha = 0.7)
                plt.ylim([0,170000])
                plt.title(f'{option_class_images3} - Mask {number_of_image3} Pixel Intensity')

                plt.subplot(133)
                plt.hist(p_v_roi1, bins = 50, color = 'purple', alpha = 0.7)
                plt.ylim([0,60000])
                plt.title(f'{option_class_images3} - ROI Pixel Intensity')

                st.pyplot(fig_images3)
                st.dataframe(pd.DataFrame({f'{option_class_images3} - Image {number_of_image3}': [round(p_v_img1.mean(), 2), round(np.median(p_v_img1), 2)], 
                                        f'{option_class_images3} - Mask {number_of_image3}': [round(p_v_mask1.mean(), 2), round(np.median(p_v_mask1), 2)], 
                                        f'{option_class_images3} - ROI': [round(p_v_roi1.mean(), 2), round(np.median(p_v_roi1), 2)]},
                                        index = ['Mean Pixel Intensity', 'Median Pixel Intensity']))

            else:
                st.write('No correct input yet or the chosen category does not have an image with this number.')

            option_class_images4 = st.selectbox('Pixel Intensities - Second Image: Choose a class', choice_of_class_images)
            number_of_image4 = st.number_input('Pixel Intensities - Second Image: Insert a number', min_value=1, value=None, placeholder="Type a number...")

            if type(number_of_image4) == int and number_of_image4 <= len(os.listdir(f'../../data/preprocessed/raw/{option_class_images4}')):
                p_v_img2, p_v_mask2, p_v_roi2 = get_pixel_intensities(option_class_images4, number_of_image4)

                fig_images4 = plt.figure(figsize=(16, 6))
                plt.subplot(131)
                plt.hist(p_v_img2, bins = 50, color = 'purple', alpha = 0.7)
                plt.ylim([0,13000])
                plt.title(f'{option_class_images4} - Image {number_of_image4} Pixel Intensity')

                plt.subplot(132)
                plt.hist(p_v_mask2, bins = 50, color = 'purple', alpha = 0.7)
                plt.ylim([0,170000])
                plt.title(f'{option_class_images4} - Mask {number_of_image4} Pixel Intensity')

                plt.subplot(133)
                plt.hist(p_v_roi2, bins = 50, color = 'purple', alpha = 0.7)
                plt.ylim([0,60000])
                plt.title(f'{option_class_images4} - ROI Pixel Intensity')

                st.pyplot(fig_images4)
                st.dataframe(pd.DataFrame({f'{option_class_images4} - Image {number_of_image4}': [round(p_v_img2.mean(), 2), round(np.median(p_v_img2), 2)], 
                                        f'{option_class_images4} - Mask {number_of_image4}': [round(p_v_mask2.mean(), 2), round(np.median(p_v_mask2), 2)], 
                                        f'{option_class_images4} - ROI': [round(p_v_roi2.mean(), 2), round(np.median(p_v_roi2), 2)]},
                                        index = ['Mean Pixel Intensity', 'Median Pixel Intensity']))

            else:
                st.write('No correct input yet or the chosen category does not have an image with this number.')


elif section == "Preprocessing":
    if selected_subtitle == "Data preparation":
        st.write("")
    elif selected_subtitle == "Region of Interests":
        st.write("")
    elif selected_subtitle == "Filters":
        st.write("")
    elif selected_subtitle == "Data Augmentation":
        st.write("")
elif section == "Interpretability":
    if selected_subtitle == "Saliency Maps":
        st.write("")
elif section == "Predictions":
    if selected_subtitle == "Model predictions":
        st.write("")

elif section == "Conclusions":
    if selected_subtitle == "Conclusion":
        st.write('#### Summary of findings')
        st.markdown('''
                    * classical ML models and DL models achieve up to 90% accuracy
                    * accuracy decreases when training on ROI images
                    * saliency maps show that lung areas from the whole images are used to classify the disease
                    * CNNs are favorable over classical ML models as they extract relevant features for interpretability and can be trained further
                    ''')

        st.write('#### Outlook')   
        st.markdown(''' 
                    * Grad-CAM for further interpretability
                    * further training on new emerging data
                    * training different CNN architecture or pre-trained models with transfer learning
                    ''')

