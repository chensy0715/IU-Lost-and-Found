import json  
import os
import numpy as np
import cv2
import time
from matplotlib import pyplot as plt

############################################################
# 4 feature point detection and 2 match method(with ransac algorithm)
		
#SIFT and flann match
def image_match(image1,image2):
	MIN_MATCH_COUNT = 10
	img1 = cv2.imread(image1,0)          # queryImage
	img2 = cv2.imread(image2,0) # trainImage
	# Initiate SIFT detector
	#sift = cv2.SIFT()
	sift= cv2.xfeatures2d.SIFT_create()
	# find the keypoints and descriptors with SIFT
	kp1, des1 = sift.detectAndCompute(img1,None)
	kp2, des2 = sift.detectAndCompute(img2,None)
	
	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks = 50)

	flann = cv2.FlannBasedMatcher(index_params, search_params)

	matches = flann.knnMatch(des1,des2,k=2)

	# store all the good matches as per Lowe's ratio test.
	good = []
	for m,n in matches:
		if m.distance < 0.7*n.distance:
			good.append(m)
			
			
			
	if len(good)>MIN_MATCH_COUNT:
		src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
		dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

		M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
		matchesMask = mask.ravel().tolist()

		h,w = img1.shape
		pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
		dst = cv2.perspectiveTransform(pts,M)

		img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
		print("good")
		return True

	else:
		print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
		matchesMask = None
		return False
		
		
		
#	if len(good)>MIN_MATCH_COUNT:
#		src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
#		dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
#
#		M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
#		matchesMask = mask.ravel().tolist()
#
#		h,w = img1.shape
#		pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
#		dst = cv2.perspectiveTransform(pts,M)
#
#		img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
#		print("good")
#
#	else:
#		print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
#		matchesMask = None
#		
		
#SURF and Flann match	
def image_match1(image1,image2):
	MIN_MATCH_COUNT = 10
	img1 = cv2.imread(image1,0)          # queryImage
	img2 = cv2.imread(image2,0) # trainImage
	# Initiate SURF detector
	#sift = cv2.SURF()
	
	surf= cv2.xfeatures2d.SURF_create()
	# find the keypoints and descriptors with SIFT
	kp1, des1 = surf.detectAndCompute(img1,None)
	kp2, des2 = surf.detectAndCompute(img2,None)
	
	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks = 50)

	flann = cv2.FlannBasedMatcher(index_params, search_params)

	matches = flann.knnMatch(des1,des2,k=2)

	# store all the good matches as per Lowe's ratio test.
	good = []
	for m,n in matches:
		if m.distance < 0.7*n.distance:
			good.append(m)
			
			
			
	if len(good)>MIN_MATCH_COUNT:
		src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
		dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

		M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
		matchesMask = mask.ravel().tolist()

		h,w = img1.shape
		pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
		dst = cv2.perspectiveTransform(pts,M)

		img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
		print("good")
		return True

	else:
		print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
		matchesMask = None
		return False
		
		
		

		
		
#ORB and flann match
def image_match2(image1,image2):
	MIN_MATCH_COUNT = 10

	img1 = cv2.imread(image1,0)          # queryImage
	img2 = cv2.imread(image2,0) # trainImage



	# Initiate ORB detector

	orb=cv2.ORB_create()
	kp1,des1=orb.detectAndCompute(img1,None)
	kp2,des2=orb.detectAndCompute(img2,None)

	#
	#FLANN_INDEX_KDTREE = 0
	#index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	#search_params = dict(checks = 50)

	
	FLANN_INDEX_LSH=6
	index_params=dict(algorithm=FLANN_INDEX_LSH, 
	                 table_number = 6, #12
	                 key_size = 12,    #20
	                 multi_probe_level = 1)#2
	search_params=dict(checks=100)


	
	
	flann = cv2.FlannBasedMatcher(index_params, search_params)

	matches = flann.knnMatch(des1,des2,k=2)

	# store all the good matches as per Lowe's ratio test.

	good = []
	for m,n in matches:
		if m.distance < 0.75*n.distance:
			good.append(m)




			
	if len(good)>MIN_MATCH_COUNT:
		src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
		dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

		M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
		matchesMask = mask.ravel().tolist()

		h,w = img1.shape
		pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
		dst = cv2.perspectiveTransform(pts,M)

		img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
		print("good")
		return True
		

	else:
		print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
		matchesMask = None
		return False
		
#brisk and flann match
def image_match3(image1,image2):
	MIN_MATCH_COUNT = 10

	img1 = cv2.imread(image1,0)          # queryImage
	img2 = cv2.imread(image2,0) # trainImage



	# Initiate brisk detector

#	orb=cv2.ORB_create()
#	kp1,des1=orb.detectAndCompute(img1,None)
#	kp2,des2=orb.detectAndCompute(img2,None)
	brisk = cv2.BRISK_create()
	kp1, des1 = brisk.detectAndCompute(img1,None)
	kp2, des2 = brisk.detectAndCompute(img2,None)


	#
	#FLANN_INDEX_KDTREE = 0
	#index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	#search_params = dict(checks = 50)
	FLANN_INDEX_LSH=6
	index_params=dict(algorithm=FLANN_INDEX_LSH, 
	                 table_number = 6, #12
	                 key_size = 12,    #20
	                 multi_probe_level = 1)#2
	search_params=dict(checks=100)


	flann = cv2.FlannBasedMatcher(index_params, search_params)

	matches = flann.knnMatch(des1,des2,k=2)

	# store all the good matches as per Lowe's ratio test.
	good = []
	for m,n in matches:
		if m.distance < 0.7*n.distance:
			good.append(m)




			
	if len(good)>MIN_MATCH_COUNT:
		src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
		dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

		M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
		matchesMask = mask.ravel().tolist()

		h,w = img1.shape
		pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
		dst = cv2.perspectiveTransform(pts,M)

		img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
		print("good")
		return True
		

	else:
		print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
		matchesMask = None
		return False
		
#orb and bf match
def image_match4(image1,image2):
	MIN_MATCH_COUNT = 10

	img1 = cv2.imread(image1,0)          # queryImage
	img2 = cv2.imread(image2,0) # trainImage



	# Initiate ORB detector

	orb=cv2.ORB_create()
	kp1,des1=orb.detectAndCompute(img1,None)
	kp2,des2=orb.detectAndCompute(img2,None)
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(des1,des2, k=2)
	# store all the good matches as per Lowe's ratio test.
	good = []
	for m,n in matches:
		if m.distance < 0.7*n.distance:
			good.append(m)

			
	if len(good)>MIN_MATCH_COUNT:
		src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
		dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

		M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
		matchesMask = mask.ravel().tolist()

		h,w = img1.shape
		pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
		dst = cv2.perspectiveTransform(pts,M)

		img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
		print("good")
		return True
		

	else:
		print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
		matchesMask = None
		return False
		
#SIFT # bf match		
def image_match5(image1,image2):
	MIN_MATCH_COUNT = 10

	img1 = cv2.imread(image1,0)          # queryImage
	img2 = cv2.imread(image2,0) # trainImage



	# Initiate SIFT detector
	sift= cv2.xfeatures2d.SIFT_create()
		# find the keypoints and descriptors with SIFT
	kp1, des1 = sift.detectAndCompute(img1,None)
	kp2, des2 = sift.detectAndCompute(img2,None)
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(des1,des2, k=2)

	# store all the good matches as per Lowe's ratio test.
	good = []
	for m,n in matches:
		if m.distance < 0.7*n.distance:
			good.append(m)

			
	if len(good)>MIN_MATCH_COUNT:
		src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
		dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

		M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
		matchesMask = mask.ravel().tolist()

		h,w = img1.shape
		pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
		dst = cv2.perspectiveTransform(pts,M)

		img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
		print("good")
		return True

	else:
		print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
		matchesMask = None
		return False
		
#SURF # bf match		
def image_match6(image1,image2):
	MIN_MATCH_COUNT = 10

	img1 = cv2.imread(image1,0)          # queryImage
	img2 = cv2.imread(image2,0) # trainImage



	# Initiate SURF detector
	surf= cv2.xfeatures2d.SURF_create()
		# find the keypoints and descriptors with SURF
	kp1, des1 = surf.detectAndCompute(img1,None)
	kp2, des2 = surf.detectAndCompute(img2,None)
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(des1,des2, k=2)

	# store all the good matches as per Lowe's ratio test.
	good = []
	for m,n in matches:
		if m.distance < 0.7*n.distance:
			good.append(m)

			
	if len(good)>MIN_MATCH_COUNT:
		src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
		dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

		M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
		matchesMask = mask.ravel().tolist()

		h,w = img1.shape
		pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
		dst = cv2.perspectiveTransform(pts,M)

		img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
		print("good")
		return True
		

	else:
		print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
		matchesMask = None
		return False
		
#brisk and bf match				
def image_match7(image1,image2):
	MIN_MATCH_COUNT = 10

	img1 = cv2.imread(image1,0)          # queryImage
	img2 = cv2.imread(image2,0) # trainImage



	# Initiate Brisk detector
	brisk = cv2.BRISK_create()
	kp1, des1 = brisk.detectAndCompute(img1,None)
	kp2, des2 = brisk.detectAndCompute(img2,None)
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(des1,des2, k=2)


	# store all the good matches as per Lowe's ratio test.
	good = []
	for m,n in matches:
		if m.distance < 0.7*n.distance:
			good.append(m)

			
	if len(good)>MIN_MATCH_COUNT:
		src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
		dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

		M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
		matchesMask = mask.ravel().tolist()

		h,w = img1.shape
		pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
		dst = cv2.perspectiveTransform(pts,M)

		img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
		print("good")
		return True
		

	else:
		print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
		matchesMask = None	
		return False	
		
#def image_match8(image1,image2):
#	MIN_MATCH_COUNT = 10
#	img1 = cv2.imread(image1,0)          # queryImage
#	img2 = cv2.imread(image2,0) 
#	
#	orb=cv2.ORB_create()
#	kp1,des1=orb.detectAndCompute(img1,None)
#	kp2,des2=orb.detectAndCompute(img2,None)
#	
#	
#	FLANN_INDEX_LSH=6
#	index_params=dict(algorithm=FLANN_INDEX_LSH, 
#	                 table_number = 6, #12
#	                 key_size = 12,    #20
#	                 multi_probe_level = 1)#2
#	search_params=dict(checks=100)
#	
#
#
#	flann = cv2.FlannBasedMatcher(index_params, search_params)
#
#	matches = flann.knnMatch(des1,des2,k=2)
#	
#	good = []
#	for m,n in matches:
#		if m.distance < 0.75*n.distance:
#			good.append(m)
#			
#	if len(good)>MIN_MATCH_COUNT:
#		src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
#		dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
#
#		M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
#		matchesMask = mask.ravel().tolist()
#
#		h,w = img1.shape
#		pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
#		dst = cv2.perspectiveTransform(pts,M)
#
#		img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
#		print("good")
#		return True
#
#	else:
#		print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
#		matchesMask = None
#		return False

#
#while(1):
#	
#	if os.path.exists('test_json.json'):
#		print(True)
#		file_in = open("test_json.json","r")
#		json_data = json.load(file_in) 
#		select_image = "1.jpg"
#		for filename in os.listdir(r"/Users/shengyuchen/Desktop/project_649/image/"):  
#			if filename != '.DS_Store':
#				src_image = '/Users/shengyuchen/Desktop/project_649/image/'+filename
#				print(filename)
#				
#		
#				if image_match(select_image,src_image):
#					json_data['LostFind']['flag'] = 'wait'
#					json_data['LostFind']['result'] = 'no found'
#					file_in.close() 
#					os.remove('test_json.json')
#					file_out = open("test1_json.json","w")
#					file_out.write(json.dumps(json_data))  
#					file_out.close()
#					
#					
#					break 
#				else:
#					print("continue")
#					continue
#			
#	
#		
##		print(json_data)
##		print("after update  --->")  
##		print(type(json_data)) 
#		
##		json_data['LostFind']['flag'] = 'wait'
##		json_data['LostFind']['result'] = 'no found'
###		print(json_data)
##		file_in.close() 
###		print("aaaaaa")
###		print(json_data)
##
##		file_out = open("test_json.json","w")
##		file_out.write(json.dumps(json_data))  
##		file_out.close() 
#
#		
#	else:
#		continue
#	time.sleep(10)



##########################################################################
#for test the time -----clear image

while(1):
	if os.path.exists('test1_json.json'):
		file_in = open("test1_json.json","r")
		json_data = json.load(file_in) 
		
		if json_data['LostFind']['flag'] == 'wait':
#			print(json_data)
			obj = json_data['LostFind']['database']['objects']
			lost_url = json_data['LostFind']['lost']['url']
			location = {}
#			print(obj)
#			print(lost_url)
#			print(type(json_data['LostFind']['database']['objects']))
			start = time.clock()
			for i in obj:
				print(i['url'])
				if image_match7(i['url'],lost_url):
					
#					print(True)
					json_data['LostFind']['flag'] = 'back'
					
					#add location tag info
#					location = i['location']
					if i["location"] not in location.values():
						location[i['name']]= i['location']
#					print(location)
					json_data['LostFind']['result'] = {'found': 'True','tag': location}
					
#					print(json_data)
					file_in.close() 
					file_out = open("test1_json.json","w")
					file_out.write(json.dumps(json_data))  
					file_out.close() 		
				else: continue
			end = time.clock()
			print(end - start)
		else:
			file_in.close() 
			
	time.sleep(10)
	
####################################################################	
	
#for test accuracy --- picture from iphone, image is nor very clear

#1
##accuracy 12/12  time: 30.64 sec per image
#start = time.clock()
##test mmouse
#image_match('/Users/shengyuchen/Desktop/image1/1.JPG',"/Users/shengyuchen/Desktop/image1/2.JPG" )
#image_match('/Users/shengyuchen/Desktop/image1/1.JPG',"/Users/shengyuchen/Desktop/image1/6.JPG" )
#image_match('/Users/shengyuchen/Desktop/image1/6.JPG',"/Users/shengyuchen/Desktop/image1/2.JPG" )
#
##test card case
#image_match('/Users/shengyuchen/Desktop/image1/3.JPG',"/Users/shengyuchen/Desktop/image1/5.JPG" )
#image_match('/Users/shengyuchen/Desktop/image1/3.JPG',"/Users/shengyuchen/Desktop/image1/10.JPG" )
#image_match('/Users/shengyuchen/Desktop/image1/10.JPG',"/Users/shengyuchen/Desktop/image1/5.JPG" )
#
## test card key
#image_match('/Users/shengyuchen/Desktop/image1/4.JPG',"/Users/shengyuchen/Desktop/image1/7.JPG" )
#image_match('/Users/shengyuchen/Desktop/image1/4.JPG',"/Users/shengyuchen/Desktop/image1/8.JPG" )
#image_match('/Users/shengyuchen/Desktop/image1/4.JPG',"/Users/shengyuchen/Desktop/image1/9.JPG" )
#image_match('/Users/shengyuchen/Desktop/image1/7.JPG',"/Users/shengyuchen/Desktop/image1/8.JPG" )
#image_match('/Users/shengyuchen/Desktop/image1/7.JPG',"/Users/shengyuchen/Desktop/image1/9.JPG" )
#image_match('/Users/shengyuchen/Desktop/image1/8.JPG',"/Users/shengyuchen/Desktop/image1/9.JPG" )
#end = time.clock()
#print(1)
#print(end - start)

#2
##accuracy 12/12 time 60.14 sec per image
#start = time.clock()
##test mmouse
#image_match1('/Users/shengyuchen/Desktop/image1/1.JPG',"/Users/shengyuchen/Desktop/image1/2.JPG" )
#image_match1('/Users/shengyuchen/Desktop/image1/1.JPG',"/Users/shengyuchen/Desktop/image1/6.JPG" )
#image_match1('/Users/shengyuchen/Desktop/image1/6.JPG',"/Users/shengyuchen/Desktop/image1/2.JPG" )
#
##test card case
#image_match1('/Users/shengyuchen/Desktop/image1/3.JPG',"/Users/shengyuchen/Desktop/image1/5.JPG" )
#image_match1('/Users/shengyuchen/Desktop/image1/3.JPG',"/Users/shengyuchen/Desktop/image1/10.JPG" )
#image_match1('/Users/shengyuchen/Desktop/image1/10.JPG',"/Users/shengyuchen/Desktop/image1/5.JPG" )
#
## test card key
#image_match1('/Users/shengyuchen/Desktop/image1/4.JPG',"/Users/shengyuchen/Desktop/image1/7.JPG" )
#image_match1('/Users/shengyuchen/Desktop/image1/4.JPG',"/Users/shengyuchen/Desktop/image1/8.JPG" )
#image_match1('/Users/shengyuchen/Desktop/image1/4.JPG',"/Users/shengyuchen/Desktop/image1/9.JPG" )
#image_match1('/Users/shengyuchen/Desktop/image1/7.JPG',"/Users/shengyuchen/Desktop/image1/8.JPG" )
#image_match1('/Users/shengyuchen/Desktop/image1/7.JPG',"/Users/shengyuchen/Desktop/image1/9.JPG" )
#image_match1('/Users/shengyuchen/Desktop/image1/8.JPG',"/Users/shengyuchen/Desktop/image1/9.JPG" )
#end = time.clock()
#print(2)
#print(end - start)

#3
##accuracy 11/12(one is not work,one image can not be detected) time: 0.97sec  per image
#start = time.clock()
##test mmouse
##image_match2('/Users/shengyuchen/Desktop/image1/1.JPG',"/Users/shengyuchen/Desktop/image1/2.JPG" )  # no work
##image_match2('/Users/shengyuchen/Desktop/image1/2.JPG',"/Users/shengyuchen/Desktop/image1/6.JPG" )
#image_match2('/Users/shengyuchen/Desktop/image1/6.JPG',"/Users/shengyuchen/Desktop/image1/2.JPG" )
#
##test card case
#image_match2('/Users/shengyuchen/Desktop/image1/3.JPG',"/Users/shengyuchen/Desktop/image1/5.JPG" )
#image_match2('/Users/shengyuchen/Desktop/image1/3.JPG',"/Users/shengyuchen/Desktop/image1/10.JPG" )
#image_match2('/Users/shengyuchen/Desktop/image1/10.JPG',"/Users/shengyuchen/Desktop/image1/5.JPG" )
#
## test card key
#image_match2('/Users/shengyuchen/Desktop/image1/4.JPG',"/Users/shengyuchen/Desktop/image1/7.JPG" )
#image_match2('/Users/shengyuchen/Desktop/image1/4.JPG',"/Users/shengyuchen/Desktop/image1/8.JPG" )
#image_match2('/Users/shengyuchen/Desktop/image1/4.JPG',"/Users/shengyuchen/Desktop/image1/9.JPG" )
#image_match2('/Users/shengyuchen/Desktop/image1/7.JPG',"/Users/shengyuchen/Desktop/image1/8.JPG" )
##image_match2('/Users/shengyuchen/Desktop/image1/7.JPG',"/Users/shengyuchen/Desktop/image1/9.JPG" )
#image_match2('/Users/shengyuchen/Desktop/image1/8.JPG',"/Users/shengyuchen/Desktop/image1/9.JPG" )
#end = time.clock()
#print(3)
#print(end - start)

#4
###accuracy 9/12(2 image can not be detected) time: 1.89 sec per image
#start = time.clock()
###test mouse
#
##image_match3('/Users/shengyuchen/Desktop/image1/1.JPG',"/Users/shengyuchen/Desktop/image1/2.JPG" ) # no work
##image_match3('/Users/shengyuchen/Desktop/image1/1.JPG',"/Users/shengyuchen/Desktop/image1/6.JPG" ) # no work
##image_match3('/Users/shengyuchen/Desktop/image1/6.JPG',"/Users/shengyuchen/Desktop/image1/2.JPG" ) # no work
#
##test card case
#image_match3('/Users/shengyuchen/Desktop/image1/3.JPG',"/Users/shengyuchen/Desktop/image1/5.JPG" )
#image_match3('/Users/shengyuchen/Desktop/image1/3.JPG',"/Users/shengyuchen/Desktop/image1/10.JPG" )
#image_match3('/Users/shengyuchen/Desktop/image1/10.JPG',"/Users/shengyuchen/Desktop/image1/5.JPG" )
#
## test card key
#image_match3('/Users/shengyuchen/Desktop/image1/4.JPG',"/Users/shengyuchen/Desktop/image1/7.JPG" )
#image_match3('/Users/shengyuchen/Desktop/image1/4.JPG',"/Users/shengyuchen/Desktop/image1/8.JPG" )
#image_match3('/Users/shengyuchen/Desktop/image1/4.JPG',"/Users/shengyuchen/Desktop/image1/9.JPG" )
#image_match3('/Users/shengyuchen/Desktop/image1/7.JPG',"/Users/shengyuchen/Desktop/image1/8.JPG" )
#image_match3('/Users/shengyuchen/Desktop/image1/7.JPG',"/Users/shengyuchen/Desktop/image1/9.JPG" )
#image_match3('/Users/shengyuchen/Desktop/image1/8.JPG',"/Users/shengyuchen/Desktop/image1/9.JPG" )
#end = time.clock()
#print(4)
#print(end - start)

#5
# acurracy is poor only 6/12 time: 0.49 sec per image
#start = time.clock()
##test mmouse
#image_match4('/Users/shengyuchen/Desktop/image1/1.JPG',"/Users/shengyuchen/Desktop/image1/2.JPG" )
#image_match4('/Users/shengyuchen/Desktop/image1/1.JPG',"/Users/shengyuchen/Desktop/image1/6.JPG" )
#image_match4('/Users/shengyuchen/Desktop/image1/6.JPG',"/Users/shengyuchen/Desktop/image1/2.JPG" )
#
##test card case
#image_match4('/Users/shengyuchen/Desktop/image1/3.JPG',"/Users/shengyuchen/Desktop/image1/5.JPG" )
#image_match4('/Users/shengyuchen/Desktop/image1/3.JPG',"/Users/shengyuchen/Desktop/image1/10.JPG" )
#image_match4('/Users/shengyuchen/Desktop/image1/10.JPG',"/Users/shengyuchen/Desktop/image1/5.JPG" )
#
## test card key
#image_match4('/Users/shengyuchen/Desktop/image1/4.JPG',"/Users/shengyuchen/Desktop/image1/7.JPG" )
#image_match4('/Users/shengyuchen/Desktop/image1/4.JPG',"/Users/shengyuchen/Desktop/image1/8.JPG" )
#image_match4('/Users/shengyuchen/Desktop/image1/4.JPG',"/Users/shengyuchen/Desktop/image1/9.JPG" )
#image_match4('/Users/shengyuchen/Desktop/image1/7.JPG',"/Users/shengyuchen/Desktop/image1/8.JPG" )
#image_match4('/Users/shengyuchen/Desktop/image1/7.JPG',"/Users/shengyuchen/Desktop/image1/9.JPG" )
#image_match4('/Users/shengyuchen/Desktop/image1/8.JPG',"/Users/shengyuchen/Desktop/image1/9.JPG" )
#end = time.clock()
#print(5)
#print(end - start)

#6
## accuracy 12/12 time: 45.69 sec per image
#start = time.clock()
##test mmouse
#image_match5('/Users/shengyuchen/Desktop/image1/1.JPG',"/Users/shengyuchen/Desktop/image1/2.JPG" )
#image_match5('/Users/shengyuchen/Desktop/image1/1.JPG',"/Users/shengyuchen/Desktop/image1/6.JPG" )
#image_match5('/Users/shengyuchen/Desktop/image1/6.JPG',"/Users/shengyuchen/Desktop/image1/2.JPG" )
#
##test card case
#image_match5('/Users/shengyuchen/Desktop/image1/3.JPG',"/Users/shengyuchen/Desktop/image1/5.JPG" )
#image_match5('/Users/shengyuchen/Desktop/image1/3.JPG',"/Users/shengyuchen/Desktop/image1/10.JPG" )
#image_match5('/Users/shengyuchen/Desktop/image1/10.JPG',"/Users/shengyuchen/Desktop/image1/5.JPG" )
#
## test card key
#image_match5('/Users/shengyuchen/Desktop/image1/4.JPG',"/Users/shengyuchen/Desktop/image1/7.JPG" )
#image_match5('/Users/shengyuchen/Desktop/image1/4.JPG',"/Users/shengyuchen/Desktop/image1/8.JPG" )
#image_match5('/Users/shengyuchen/Desktop/image1/4.JPG',"/Users/shengyuchen/Desktop/image1/9.JPG" )
#image_match5('/Users/shengyuchen/Desktop/image1/7.JPG',"/Users/shengyuchen/Desktop/image1/8.JPG" )
#image_match5('/Users/shengyuchen/Desktop/image1/7.JPG',"/Users/shengyuchen/Desktop/image1/9.JPG" )
#image_match5('/Users/shengyuchen/Desktop/image1/8.JPG',"/Users/shengyuchen/Desktop/image1/9.JPG" )
#end = time.clock()
#print(6)
#print(end - start)

#7
## accurcy 12/12  time: 76.24 secper image
#start = time.clock()
##test mmouse
#image_match6('/Users/shengyuchen/Desktop/image1/1.JPG',"/Users/shengyuchen/Desktop/image1/2.JPG" )
#image_match6('/Users/shengyuchen/Desktop/image1/1.JPG',"/Users/shengyuchen/Desktop/image1/6.JPG" )
#image_match6('/Users/shengyuchen/Desktop/image1/6.JPG',"/Users/shengyuchen/Desktop/image1/2.JPG" )
#
##test card case
#image_match6('/Users/shengyuchen/Desktop/image1/3.JPG',"/Users/shengyuchen/Desktop/image1/5.JPG" )
#image_match6('/Users/shengyuchen/Desktop/image1/3.JPG',"/Users/shengyuchen/Desktop/image1/10.JPG" )
#image_match6('/Users/shengyuchen/Desktop/image1/10.JPG',"/Users/shengyuchen/Desktop/image1/5.JPG" )
#
## test card key
#image_match6('/Users/shengyuchen/Desktop/image1/4.JPG',"/Users/shengyuchen/Desktop/image1/7.JPG" )
#image_match6('/Users/shengyuchen/Desktop/image1/4.JPG',"/Users/shengyuchen/Desktop/image1/8.JPG" )
#image_match6('/Users/shengyuchen/Desktop/image1/4.JPG',"/Users/shengyuchen/Desktop/image1/9.JPG" )
#image_match6('/Users/shengyuchen/Desktop/image1/7.JPG',"/Users/shengyuchen/Desktop/image1/8.JPG" )
#image_match6('/Users/shengyuchen/Desktop/image1/7.JPG',"/Users/shengyuchen/Desktop/image1/9.JPG" )
#image_match6('/Users/shengyuchen/Desktop/image1/8.JPG',"/Users/shengyuchen/Desktop/image1/9.JPG" )
#end = time.clock()
#print(7)
#print(end - start)

#8
## acurracy 11/12 time 10.52 sec per image
#start = time.clock()
##test mmouse
#image_match7('/Users/shengyuchen/Desktop/image1/1.JPG',"/Users/shengyuchen/Desktop/image1/2.JPG" )
#image_match7('/Users/shengyuchen/Desktop/image1/1.JPG',"/Users/shengyuchen/Desktop/image1/6.JPG" )
#image_match7('/Users/shengyuchen/Desktop/image1/6.JPG',"/Users/shengyuchen/Desktop/image1/2.JPG" )
#
##test card case
#image_match7('/Users/shengyuchen/Desktop/image1/3.JPG',"/Users/shengyuchen/Desktop/image1/5.JPG" )
#image_match7('/Users/shengyuchen/Desktop/image1/3.JPG',"/Users/shengyuchen/Desktop/image1/10.JPG" )
#image_match7('/Users/shengyuchen/Desktop/image1/10.JPG',"/Users/shengyuchen/Desktop/image1/5.JPG" )
#
## test card key
#image_match7('/Users/shengyuchen/Desktop/image1/4.JPG',"/Users/shengyuchen/Desktop/image1/7.JPG" )
#image_match7('/Users/shengyuchen/Desktop/image1/4.JPG',"/Users/shengyuchen/Desktop/image1/8.JPG" )
#image_match7('/Users/shengyuchen/Desktop/image1/4.JPG',"/Users/shengyuchen/Desktop/image1/9.JPG" )
#image_match7('/Users/shengyuchen/Desktop/image1/7.JPG',"/Users/shengyuchen/Desktop/image1/8.JPG" )
#image_match7('/Users/shengyuchen/Desktop/image1/7.JPG',"/Users/shengyuchen/Desktop/image1/9.JPG" )
#image_match7('/Users/shengyuchen/Desktop/image1/8.JPG',"/Users/shengyuchen/Desktop/image1/9.JPG" )
#end = time.clock()
#print(8)
#print(end - start)

		
