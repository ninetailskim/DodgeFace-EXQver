import paddlehub as hub
import cv2
import numpy as np
# import pygame as pg
import time
import random
import os
import math
import copy
import glob
from ffpyplayer.player import MediaPlayer
from PIL import Image, ImageDraw, ImageFont
import emoji

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

op = "resources/op.mp4"

currentTime = 0
lastTime = -1
genTime = [5, 10, 20]
currentIndex = 0
gm = [0, 7, 0]
W = 0
H = 0
showimg = None
checkimg = None
minPIXEL = 2500
dangerousPIXEL = 4500
balls = []
scale = 2
llock = False

class detUtils():
    def __init__(self):
        super(detUtils, self).__init__()
        self.lastres = None
        self.module = hub.Module(name="ultra_light_fast_generic_face_detector_1mb_320")
    
    def distance(self, a, b):
        return math.sqrt(math.pow(a[0]-b[0], 2) + math.pow(a[1]-b[1], 2))

    def iou(self, bbox1, bbox2):

        b1left = bbox1['left']
        b1right = bbox1['right']
        b1top = bbox1['top']
        b1bottom = bbox1['bottom']

        b2left = bbox2['left']
        b2right = bbox2['right']
        b2top = bbox2['top']
        b2bottom = bbox2['bottom']

        area1 = (b1bottom - b1top) * (b1right - b1left)
        area2 = (b2bottom - b2top) * (b2right - b2left)

        w = min(b1right, b2right) - max(b1left, b2left)
        h = min(b1bottom, b2bottom) - max(b1top, b2top)

        dis = self.distance([(b1left+b1right)/2, (b1bottom+b1top)/2],[(b2left+b2right)/2, (b2bottom+b2top)/2])

        if w <= 0 or h <= 0:
            return 0, dis
        
        iou = w * h / (area1 + area2 - w * h)
        return iou, dis


    def dodet(self, frame):
        result = self.module.face_detection(images=[frame], use_gpu=True)
        result = result[0]['data']
        if isinstance(result, list):
            if len(result) == 0:
                return None, None
            if len(result) > 1:
                if self.lastres is not None:
                    maxiou = -float('inf')
                    maxi = 0
                    mind = float('inf')
                    mini = 0
                    for index in range(len(result)):
                        tiou, td = self.iou(self.lastres, result[index])
                        if tiou > maxiou:
                            maxi = index
                            maxiou = tiou
                        if td < mind:
                            mind = td
                            mini = index  
                    if tiou == 0:
                        return result[mini], result
                    else:
                        return result[maxi], result
                else:
                    self.lastres = result[0]
                    return result[0], result
            else:
                self.lastres = result[0]
                return result[0], result
        else:
            return None, None
           
def resize(img, size):
    h, w = img.shape[:2]
    ml = max(h, w)
    t = ml / size
    return cv2.resize(img, (int(w / t), int(h / t)))

def cv2ImgAddText(img, text, left, top, textColor=(255, 0, 0), textSize=50):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    draw = ImageDraw.Draw(img)

    fontStyle = ImageFont.truetype(
        "font/simsun.ttc", textSize, encoding="utf-8")

    draw.text((left+1, top+1), text, (0, 0, 0), font=fontStyle)
    draw.text((left, top), text, textColor, font=fontStyle)

    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

def pngmergealpha(imgpath, dsize):
    img = cv2.imread(imgpath, cv2.IMREAD_UNCHANGED)
    rimg = resize(img, dsize)
    mask = rimg[:,:,3]
    mask[mask > 0] = 1
    return rimg[:,:,:-1] * np.repeat(mask[:,:,np.newaxis], 3, 2)

def loadResources(dsize=100, tsize=50):
    dfiles = glob.glob("resources/d/*.png")
    tfiles = glob.glob("resources/t/*.png")
    dimgs = []
    timgs = []
    dimgs = [pngmergealpha(df, dsize) for df in dfiles]
    timgs = [pngmergealpha(tf, tsize) for tf in tfiles]

    return dimgs, timgs       

def getPIXEL(x, y, w, h):
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)
    t = y - h if y - h > 0 else 0
    l = x - w if x - w > 0 else 0
    b = y + h if y + h < H else H - 1
    r = x + w if x + w < W else W - 1
    tt = 0 if y - h > 0 else h - y
    tl = 0 if x - w > 0 else w - x
    tb = 2 * h if y + h < H else h - y + H - 1
    tr = 2 * w if x + w < W else w - x + W - 1
    return int(t), int(l), int(b), int(r), int(tt), int(tl), int(tb), int(tr)

class Skill():
    def __init__(self, interval, gm):
        self.stime = 0
        self.interval = interval
        self.gm = gm
        self.finish = False

    def trigger(self):
        self.stime = time.time()
        self.play()

    def play(self):
        pass

class Balloon(Skill):
    def __init__(self, interval, gm):
        super(Balloon, self).__init__(interval, gm)

    def play(self):
        # print("Balloon Play")
        if self.finish is False:
            self.gm.glive()
            if np.floor(time.time() - self.stime) >= self.interval:
                self.finish = True

class Lock(Skill):
    def __init__(self, interval, gm):
        super(Lock, self).__init__(interval, gm)

    def play(self):
        global llock
        #print("Lock Play")
        if self.finish is False:
            llock = True
            if np.floor(time.time() - self.stime) >= self.interval:
                self.finish = True
                llock = False
        #print("Lock Play end:", llock)

class Tansir(Skill):
    def __init__(self,interval, gm):
        super(Tansir, self).__init__(interval, gm)

    def play(self):
        # print("Tansir Play")
        if self.finish is False:
            self.gm.nlive()
            if np.floor(time.time() - self.stime) >= self.interval:
                self.finish = True

class GameManager():
    def __init__(self, lives):
        super(GameManager, self).__init__()
        self.lives = lives
        self.skill = []

    def nlive(self):
        self.lives -= 1
    
    def glive(self):
        self.lives += 1

    def appendskill(self, skill):
        self.skill.append(skill)

    def play(self):
        if len(self.skill) > 0:
            for s in self.skill:
                if s.finish:
                    self.skill.remove(s)
                else:
                    s.play()

class Ball():
    
    x = None
    y = None
    speed_x = None
    speed_y = None

    def __init__(self, x, y, speed_x, speed_y, img, skill):
        self.x = x
        self.y = y
        self.speed_x = speed_x
        self.speed_y = speed_y
        self.img = img
        mask = np.zeros_like(img)
        mask[img > 0] = 1
        self.mask = mask
        self.h, self.w = img.shape[:2]  
        self.skill = skill           

    def move(self, screen, checkimg):
        global GM
        global llock
        # print(llock)
        if not llock:
            self.x += self.speed_x
            self.y += self.speed_y
        
            if self.x > W - self.w/2 or self.x < self.w/2:
                self.speed_x = -self.speed_x

            if self.y > H - self.h/2 or self.y < self.h/2:
                self.speed_y = -self.speed_y

        t, l, b, r, tt, tl, tb, tr = getPIXEL(self.x, self.y, self.w/2, self.h/2)

        ctimg = checkimg[t:b,l:r]  
        stimg = screen[t:b,l:r]          
        
        if np.sum(ctimg[self.mask[tt:tb,tl:tr]>0]) > 0:
            self.skill.trigger()
            if self.skill.finish is False:
                GM.appendskill(self.skill)
            return True
        else:
            screen[t:b,l:r] = screen[t:b,l:r] * (1 - self.mask[tt:tb,tl:tr]) +  self.mask[tt:tb,tl:tr] * self.img[tt:tb,tl:tr]
            return False

def randomXY(h, w):
    x = 0
    y = 0
    while x < w/2 or x > W-w/2-1:
        x = random.randint(0, W-1)
    while y < h/2 or y > H-h/2-1:
        y = random.randint(0, H-1)
    return x, y

def inseg(x, y, w,h):
    global checkimg
    if checkimg is None:
        return False
    else:
        t,l,b,r,_,_,_,_ = getPIXEL(x, y, w+200,h+200)
        if np.sum(checkimg[t:b,l:r]) > 0:
            return True
        else:
            return False

def PlayVideo(video_path, H, W):
    video=cv2.VideoCapture(video_path)
    player = MediaPlayer(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    sleep_ms = int(np.round((1/fps)*1000))
    while True:
        grabbed, frame = video.read()
        audio_frame, val = player.get_frame()
        if not grabbed:
            break
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break
        cv2.imshow("Game", cv2.resize(frame,(W, H)))
        if val != 'eof' and audio_frame is not None:
            #audio
            img, t = audio_frame
    video.release()
    cv2.destroyAllWindows()

GM = GameManager(1)

def create_ball(timgs):
    global GM
    img = timgs[random.randint(0, len(timgs)-1)]
    h,w = img.shape[:2]
    
    speed_x = 0
    speed_y = 0
    while speed_x == 0 or speed_y == 0:
        speed_x = random.randint(-5, 5)
        speed_y = random.randint(-5, 5)

    x, y = randomXY(h, w)
    if inseg(x,y,w/2,h/2):
        x, y = randomXY(h, w)
    ratio = random.randint(1, 100)
    if ratio < 15:
        ratio = random.randint(1, 100)
        if ratio < 70:
            b = Ball(x, y, speed_x, speed_y, pngmergealpha("resources/s/balloon.png", 50), Balloon(0, GM))
        else:
            b = Ball(x, y, speed_x, speed_y, pngmergealpha("resources/s/lock.png", 50), Lock(random.randint(1,10), GM))
    else:
        b = Ball(x, y, speed_x, speed_y, img, Tansir(0, GM))
    balls.append(b) 

def ball_manager(timgs):
    global showimg
    global checkimg
    global currentIndex
    global currentTime
    global lastTime
    global gm
    global genTime
    global GM
    if currentTime != lastTime:
        if currentIndex < len(gm):
            if currentTime < genTime[currentIndex]:
                for i in range(gm[currentIndex]):
                    create_ball(timgs)
            else:
                currentIndex += 1
                if currentIndex >= len(gm):
                    currentIndex = len(gm) - 1
        
        lastTime = currentTime

    for b in balls:
        if b.move(showimg, checkimg):
            balls.remove(b)
    if GM.lives <= 0:
        return True
    return False

def loadMap(H, W):
    mapp = cv2.imread("resources/map.png")
    return cv2.resize(mapp, (W, H))

def main():
    global showimg
    global H
    global W
    global currentTime
    global startTime
    global balls
    global currentIndex
    global checkimg
    global GM
    restart = True
    du = detUtils()
    videostream = "test.mp4"
    # videostream = 0
    cap = cv2.VideoCapture(videostream)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if max(H, W) < 1000:
        H, W = H * scale, W * scale
    showimgt = loadMap(H, W)
    dimgs, timgs = loadResources(100, 50)
    ret, frame = cap.read()
    PlayVideo(op, H, W)
    dimgindex = 0
    while restart:
        restart = False
        lastTime = -1
        showimg = None
        checkimg = None
        currentIndex = 0
        balls = []
        startTime = time.time()
        lastres = None
        llt = 0
        GM = GameManager(1)
        
        while True:
            
            ret, frame = cap.read()
            if videostream == 0:
                frame = cv2.flip(frame, 1)

            cv2.imshow("self", frame)

            if ret == True:

                frame = cv2.resize(frame, (W, H))
                currentTime = math.floor(time.time() - startTime)
                
                if currentTime % 4 == 0 and currentTime != llt:
                    dimgindex = np.random.randint(0,len(dimgs)-1)
                    llt = currentTime
                dimg = copy.deepcopy(dimgs[dimgindex])
                
                h,w = dimg.shape[:2]
                
                detres, ress = du.dodet(frame)
                GM.play()
                if detres is not None:                    
                    lastres = detres

                    # showimg = np.ones_like(frame) * 255
                    showimg = copy.deepcopy(showimgt)
                    # for res in ress:
                    #     cv2.rectangle(showimg,(int(res['left']), int(res['top'])),(int(res['right']), int(res['bottom'])),(0,0,255),5)
                    checkimg = np.zeros_like(frame)
                    top = detres['top']
                    right = detres['right']
                    left = detres['left']
                    bottom = detres['bottom']
                    # cv2.rectangle(showimg,(int(detres['left']), int(detres['top'])),(int(detres['right']), int(detres['bottom'])),(255,0,0),5)
                    t, l, b, r, tt, tl, tb, tr = getPIXEL((right+left)/2, (top+bottom)/2, w/2,h/2)
                    
                    dimg = dimg[tt:tb,tl:tr]
                    mask = np.zeros_like(dimg)
                    mask[dimg>0] = 1
                    # cv2.imshow("ttt", dimg)
                    showimg[t:b,l:r] = showimg[t:b,l:r] * (1 - mask) + dimg
                    checkimg[t:b,l:r] = checkimg[t:b,l:r] + mask
                    if r < 0 or b < 0 or t >= H or l >= W:
                        gameover = True
                    else:
                        gameover = False
                    if not gameover:
                        gameover = ball_manager(timgs)
                    showimg = showimg.astype(np.uint8)
                    
                    # showimg = cv2.putText(showimg, "Lives: " + heart + "Pixel: %d"% int((right - left) * (bottom - top))+" Time: %.2f"% (time.time() - startTime), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 3)
                    showimg = cv2ImgAddText(showimg, "Lives: %d" %GM.lives + "Pixel: %d"% int((right - left) * (bottom - top))+" Time: %.2f"% (time.time() - startTime), 0, 10)
                    if gameover:
                        # showimg = cv2.putText(showimg, "You Lose! Time: %2f" % (time.time() - startTime), (0, int(H/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
                        showimg = cv2ImgAddText(showimg, "You Lose! Time: %.2f" % (time.time() - startTime), 0, int(H/2))
                        cv2.imshow('Game', showimg)
                        if cv2.waitKey(0) == ord('r'):
                            restart = True
                        break
                    else:
                        cv2.imshow('Game', showimg)
                        cv2.waitKey(1)
                else:
                    if lastres is None:
                        if showimg is None:
                            # showimg = np.ones_like(frame) * 255
                            showimg = copy.deepcopy(showimgt)
                        # showimg = cv2.putText(showimg, "Keep your face in camera", (0, int(H/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 3)
                        showimg = cv2ImgAddText(showimg, "Keep your face in camera", 0, int(H/2))
                        cv2.imshow('Game', showimg)
                        if cv2.waitKey(0) == ord('r'):
                            restart = True
                        break
                    else:
                        detres = lastres
                        # showimg = np.ones_like(frame) * 255
                        showimg = copy.deepcopy(showimgt)
                        checkimg = np.zeros_like(frame)
                        top = detres['top']
                        right = detres['right']
                        left = detres['left']
                        bottom = detres['bottom']
                        
                        t, l, b, r, tt, tl, tb, tr = getPIXEL((right+left)/2, (top+bottom)/2, w/2,h/2)
                        
                        dimg = dimg[tt:tb,tl:tr]
                        mask = np.zeros_like(dimg)
                        mask[dimg>0] == 1
                        showimg[t:b,l:r] = showimg[t:b,l:r] * (1 - mask) + dimg
                        checkimg[t:b,l:r] = checkimg[t:b,l:r] + mask
                        
                        gameover = ball_manager(timgs)
                        showimg = showimg.astype(np.uint8)
                        
                        
                        # showimg = cv2.putText(showimg, "Lives: " + heart + "Pixel: %d"% int((right - left) * (bottom - top))+" Time: %.2f"% (time.time() - startTime), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 3)
                        showimg = cv2ImgAddText(showimg, "Lives: %d" %GM.lives + "Pixel: %d"% int((right - left) * (bottom - top))+" Time: %.2f"% (time.time() - startTime), 0, 10)
                        if gameover:
                            #showimg = cv2.putText(showimg, "You Lose! Time: %2f" % (time.time() - startTime), (0, int(H/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
                            showimg = cv2ImgAddText(showimg, "You Lose! Time: %.2f" % (time.time() - startTime), 0, int(H/2))
                            cv2.imshow('Game', showimg)
                            if cv2.waitKey(0) == ord('r'):
                                restart = True
                            break
                        else:
                            cv2.imshow('Game', showimg)
                            cv2.waitKey(1)
            else:
                if showimg is None:
                    # showimg = np.ones_like(frame) * 255
                    showimg = copy.deepcopy(showimgt)
                # showimg = cv2.putText(showimg, "Check your camera!", (0, int(H/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 3)
                showimg = cv2ImgAddText(showimg, "Check your camera!", 0, int(H/2))
                cv2.imshow('Game', showimg)
                if cv2.waitKey(0) == ord('r'):
                    restart = True
                break
    cv2.destroyAllWindows()
    cap.release()

if __name__ == '__main__':
    main()