# https://www.google.co.jp/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwj10-u1k-r0AhXPdd4KHadOA8oQwqsBegQIBBAB&url=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3DzyzIgB4H06c&usg=AOvVaw0rV_5kFli1-LbhtTjvxCuw
# https://www.google.co.jp/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwj10-u1k-r0AhXPdd4KHadOA8oQwqsBegQIAhAB&url=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3DSXJlQ9w2gqs&usg=AOvVaw1YG7wdSFeSWblj-Vuqpkie

import tensorflow.keras
import cv2
import numpy as np  
import time

from selenium import webdriver
from selenium.webdriver.chrome import service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import chromedriver_binary

#def send_msg(msg, driver, delay=1):
def send_msg(msg, delay=1):
    # to input form 
    driver.find_element(By.ID, "send_text").send_keys(msg+"$")
    # to click button 
    driver.find_element(By.ID, "send").click()
    # to remove input form
    driver.find_element(By.ID, "send_text").clear()
    time.sleep(delay)

def disconnect_bl():
    driver.find_element(By.ID, "send_text").send_keys("s"+"$")
    driver.find_element(By.ID, "connect").click()

# file path ##########################################################################
model_dir = "./model/"
model_file = model_dir + "keras_model.h5"
labels_file = model_dir + "labels.txt"

#send_bl_url = "file:///C:/Users/kamek/ml_car1/send_bl.html"
send_bl_url = "file:///Users/sorasato/github/mlcar_microbit/send_bl.html"
driver_path = 'driver/chromedriver'

# model settings #####################################################################

np.set_printoptions(suppress=True) # define display style
model = tensorflow.keras.models.load_model(model_file) # Load model
with open(labels_file, "r") as f:
    class_names = f.read().split("\n")

# bluetooth setting and on browser ####################################################

options = Options()
options.add_experimental_option('excludeSwitches', ['enable-logging'])
options.use_chromium = True



print('connectiong to remote browser...')
# chrome_service = service.Service(executable_path=driver_path)
# driver = webdriver.Chrome(service=chrome_service, options=options)
driver = webdriver.Chrome()
# driver = webdriver.Chrome(executable_path=driver_path)
# chrome_service = service.Service(ChromeDriverManager().install())
# driver = webdriver.Chrome(service=chrome_service)

# action for browser
driver.get(send_bl_url)
print(driver.current_url)

# to connect
#driver.find_element(By.ID, "connect").click()
time.sleep(10)

# define message var ###############################################################

# msg -> f: forward, s: stop, l: left, r: right
msg=""
label_list = ["0 front", "1 right", "2 left", "3 stop"]

# for  capture ##############################################################
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
size = (224,224)

# delay for wholes #######################################################
time.sleep(5)

# on capture ##############################################################
cap = cv2.VideoCapture(0)




try:
    while cap.isOpened():
        time.sleep(0.001)
        start = time.time()

        ret, img = cap.read()
        height, width, channels = img.shape
        scale_value = width/height
        img_resized = cv2.resize(img, size, fx=scale_value, fy=1, interpolation=cv2.INTER_NEAREST)
        img_array = np.asarray(img_resized)
        normalized_img_array = (img_array.astype(np.float32)/ 127.0) - 1
        data[0] = normalized_img_array

        # prediction
        prediction = model.predict(data)

        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        end = time.time()
        totalTime = end - start

        fps = 1 / totalTime

        # you maybe need on #############################################################
        #cv2.putText(img, f"FPS: {int(fps)}", (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
        #cv2.putText(img, class_name, (75,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
        #cv2.putText(img, str(float("{:.2f}".format(confidence_score*100))) + "%", (75,100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
    
        #cv2.imshow("Classification Resized", img_resized)
        cv2.imshow("Classification Original", img)
        #################################################################################
        print(class_name)
        
        # send msg via bluetooth
        if class_name == label_list[0]:
            send_msg("f")
        elif class_name == label_list[2]:
            send_msg("r")
        elif class_name == label_list[1]:
            send_msg("l")
        elif class_name == label_list[3]:
            send_msg("s")
    
except KeyboardInterrupt:
    # end 
    # for capture
    cap.release()
    print("capture release!!")
    # for bluetooth
    disconnect_bl()
    print("disconnected !")
    driver.quit()
    print("browser quit")
    print("deactivate")
