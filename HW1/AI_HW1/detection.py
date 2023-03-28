from configparser import Interpolation
import os
import cv2
import matplotlib.pyplot as plt

def detect(dataPath, clf):
    """
    Please read detectData.txt to understand the format. Load the image and get
    the face images. Transfer the face images to 19 x 19 and grayscale images.
    Use clf.classify() function to detect faces. Show face detection results.
    If the result is True, draw the green box on the image. Otherwise, draw
    the red box on the image.
      Parameters:
        dataPath: the path of detectData.txt
      Returns:
        No returns.
    """
    # Begin your code (Part 4)
    # raise NotImplementedError("To be implemented")
    """
    define red and green in BGR form.
    use readline to read txt file in line, get img name and number of faces. read img 2 times, one in grayscale.
    crop the img where face locates, resize it to a 19x19 square and classify the square.
    if a face is detected, draw a green rectangle on the full colored img, else draw a red one.
    convert the BGR img to RGB img and show. 
    """
    f = open(dataPath,'r')
    red = (0,0,255)
    green  = (0,255,0)

    for a in range(2):
      correct = 0
      line = f.readline()
      str = line.split()
      img = cv2.imread(os.path.join('data','detect',str[0]),cv2.IMREAD_GRAYSCALE)
      color_img = cv2.imread(os.path.join('data','detect',str[0]))
      for i in range(int(str[1])):
        l = f.readline()
        s = l.split()
        pic = img[int(s[1]):int(s[1])+int(s[3]),int(s[0]):int(s[0])+int(s[2])]
        pic = cv2.resize(pic, (19, 19),interpolation=cv2.INTER_NEAREST)
        find = clf.classify(pic)
        if(find == 1):
          correct +=1
          cv2.rectangle(color_img,(int(s[0]),int(s[1])),(int(s[0])+int(s[3]),int(s[1])+int(s[2])),green,2)
        else:
          cv2.rectangle(color_img,(int(s[0]),int(s[1])),(int(s[0])+int(s[3]),int(s[1])+int(s[2])),red,2)

      print('find',correct,'faces in total')
      b,g,r = cv2.split(color_img)
      new_img = cv2.merge([r,g,b])
      fig, ax = plt.subplots(1, 1)
      ax.axis('off')
      ax.set_title(str[0])
      ax.imshow(new_img)
      plt.show() 
    f.close()
    # End your code (Part 4)
