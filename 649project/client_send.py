import json  
import os
import numpy as np
import cv2
import time
from matplotlib import pyplot as plt

name = input("input the things name: ")
url = input("input the rount: ")
file_name = input("input the file name: ")
#print(file_name)



if os.path.exists(file_name):
	file_in = open(file_name,"r")
	json_data = json.load(file_in) 
	if json_data['LostFind']['flag'] == 'wait':
		json_data['LostFind']['flag'] = 'back'
	json_data['LostFind']['lost'] = {'name': name,"url":url}
	json_data['LostFind']['result'] = {'found': 'Null','tag': "Null"}
	file_in.close() 
	file_out = open(file_name,"w")
	file_out.write(json.dumps(json_data))  
	file_out.close() 
