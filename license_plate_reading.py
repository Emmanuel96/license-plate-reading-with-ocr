import pytesseract 
import matplotlib.pyplot as plt
import cv2
import glob
import os 
import numpy as np

import wget, zipfile, os
filename = 'license-plates'

# if not os.path.isfile(filename): 
#     filename = wget.download('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/CV0101/Dataset/license-plates.zip')
#     with zipfile.ZipFile("license-plates.zip", "r") as zip_ref: 
#         zip_ref.extractall()
# pytesseract.pytesseract.tesseract_cmd = r'<full_path_to_your_tesseract_executable>'

path_for_license_plates = os.getcwd() + "/license-plates/**/*.jpg"
list_license_plates = []
predicted_license_plates = []

for path_to_license_plate in glob.glob(path_for_license_plates, recursive=True):
    
    license_plate_file = path_to_license_plate.split("/")[-1]
    license_plate, _ = os.path.splitext(license_plate_file)
    # print(license_plate)
    '''
    Here we append the actual license plate to a list
    '''
    list_license_plates.append(license_plate)
    
    '''
    Read each license plate image file using openCV
    '''
    img = cv2.imread(path_to_license_plate)
    
    '''
    We then pass each license plate image file to the Tesseract OCR engine using 
    the Python library wrapper for it. We get back a predicted_result for the license plate.
    We append the predicted_result in a list and compare it with the original the license plate
    '''
    predicted_result = pytesseract.image_to_string(img, lang='eng',config='--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    
    filter_predicted_result = "".join(predicted_result.split()).replace(":", "").replace("-", "")
    predicted_license_plates.append(filter_predicted_result)

print("Actual License Plate", "\t", "Predicted License Plate", "\t", "Accuracy")
print("--------------------", "\t", "-----------------------", "\t", "--------")

def calculate_predicted_accuracy(actual_list, predicted_list):
    for actual_plate, predict_plate in zip(actual_list, predicted_list):
        accuracy = "0%"
        num_matches = 0
        if actual_plate == predict_plate:
            accuracy = "100%"
        else:
            if len(actual_plate) == len(predict_plate):
                for a, p in zip(actual_plate, predict_plate):
                    if a == p:
                        num_matches += 1
                accuracy = str(round((num_matches/len(actual_plate)), 2) * 100)
                accuracy += "%"
        print("     ", actual_plate, "\t\t\t", predict_plate, "\t\t  ", accuracy)

        
# calculate_predicted_accuracy(list_license_plates, predicted_license_plates)

# 1.1 Read in the license plate file of GWT2180
test_license_plate = cv2.imread(os.getcwd() + "/license-plates/GWT2180.jpg")
# plt.imshow(test_license_plate)
# plt.axis('off'
# plt.title('GWT2180 license plate')
# plt.show()



resize_test_license_plate = cv2.resize(test_license_plate, None, fx=1.3, fy=1.3, interpolation=cv2.INTER_LANCZOS4)
# next we convert our image into a grey scale image 
grayscale_resize_test_license_plate = cv2.cvtColor(resize_test_license_plate, cv2.COLOR_BGR2GRAY)
# next up we apply a gaussian blur to the grey scale image 
gaussian_blur_license_plate = cv2.GaussianBlur(grayscale_resize_test_license_plate, (1,3), 0, cv2.BORDER_CONSTANT)

plt.imshow(gaussian_blur_license_plate)
# plt.imshow(dst)

plt.axis('off')
plt.title('GWT2180 license plate')
# plt.show()
# next we pass our image to see the result 
new_predicted_result_GWT2180 = pytesseract.image_to_string(gaussian_blur_license_plate, lang='eng',
config='--oem 3 -l eng --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
filter_new_predicted_result_GWT2180 = "".join(new_predicted_result_GWT2180.split()).replace(":", "").replace("-", "")
print(filter_new_predicted_result_GWT2180)


# 1.1 Read in the license plate file of JSQ1413
# Write your code below:

# Read the license plate file and display it
# test_license_plate = cv2.imread(os.getcwd() + "/license-plates/JSQ1413.jpg")
# plt.imshow(test_license_plate)
# plt.axis('off')
# plt.title('GWT2180 license plate')
# plt.show()

# # 1.2 Apply the image processing techniques to the license plate of JSQ1413 described above 
# # Write your code below:
# resize_test_license_plate = cv2.resize(test_license_plate, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
# # Image resize
# grayscale_resize_test_license_plate = cv2.cvtColor(resize_test_license_plate, cv2.COLOR_BGR2GRAY)
# # Denoising the image
# gaussian_blur_license_plate = cv2.GaussianBlur(grayscale_resize_test_license_plate, (5, 5), 0)

# # 1.3 Pass the modified license plate file to the Tesseract Engine. Report your findings 
# # Write your code below:
# new_predicted_result_GWT2180 = pytesseract.image_to_string(gaussian_blur_license_plate, lang='eng',
# config='--oem 3 -l eng --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
# filter_new_predicted_result_GWT2180 = "".join(new_predicted_result_GWT2180.split()).replace(":", "").replace("-", "")
# print(filter_new_predicted_result_GWT2180)