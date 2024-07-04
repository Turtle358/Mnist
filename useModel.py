import pygame, cv2
from pygame.locals import *
import numpy as np
import tensorflow as tf
winsize_x = 640
winsize_y = 480
white = (255,255,255)
black = (0,0,0)
red = (255,0,0)
img_cnt = 1
Boundry = 5
Predict = True
imagesave = False
model = tf.keras.models.load_model("./model.keras")
Labels = {0:"Zero",1:'One',
          2:'Two',3:"Three",
          4:'Four',5:'Five',
          6:'Six',7:'Seven',
          8:'Eight',9:'Nine'}
iswriting = False
number_xcord = []
number_ycord = []
# Initialise pygame
pygame.init()
font = pygame.font.Font('arial.ttf',18)
DISPLAYSURFACE = pygame.display.set_mode((winsize_x,winsize_y))
pygame.display.set_caption('Digit Board')
while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            exit()
        if event.type == MOUSEMOTION and iswriting:
            xcord,ycord = event.pos
            pygame.draw.circle(DISPLAYSURFACE,white,(xcord,ycord),4,0)
            number_xcord.append(xcord)
            number_ycord.append(ycord)
        if event.type == MOUSEBUTTONDOWN:
            iswriting = True
        if event.type == MOUSEBUTTONUP:
            iswriting = False
            number_xcord = sorted(number_xcord)
            number_ycord = sorted(number_ycord)

            rect_min_x, rect_max_x = max(number_xcord[0]-Boundry,0),min(winsize_x,number_xcord[-1]+Boundry)
            rect_min_y, rect_max_y = max(number_ycord[0]-Boundry, 0), min(winsize_y, number_ycord[-1] + Boundry)

            number_xcord = []
            number_ycord = []
            img_arr = np.array(pygame.PixelArray(DISPLAYSURFACE))[rect_min_x:rect_max_y,rect_min_y:rect_max_y].T.astype(np.float32)
            if imagesave:
                cv2.imwrite('image.png')
                img_cnt +=1
            if Predict:
                image = cv2.resize(img_arr,(28,28))
                image = np.pad(image,(10,10),'constant',constant_values=0)
                image = cv2.resize(image,(28,28))/255

                label = str(Labels[np.argmax(model.predict(image.reshape(1,28,28,1)))])

                textSurface = font.render(label, True,red,white)
                textRectObj = textSurface.get_rect()
                textRectObj.left, textRectObj.bottom = rect_min_x,rect_min_y

                DISPLAYSURFACE.blit(textSurface,textRectObj)
        if event.type == KEYDOWN:
            if event.unicode == 'n':
                DISPLAYSURFACE.fill(black)
    pygame.display.update()
