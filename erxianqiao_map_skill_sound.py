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
import argparse
import _thread
import sound

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
level = 5

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
           
class Resources():
    def __init__(self, dsize, tsize, ssize):
        self.dsize = dsize
        self.tsize = tsize
        self.ssize = ssize
        self.init()

    def init(self):
        dfiles = glob.glob("resources/d/*.png")
        tfiles = glob.glob("resources/t/*.png")
        dimgs = []
        timgs = []
        self.dimgs = [self.pngmergealpha(df, self.dsize) for df in dfiles]
        self.timgs = [self.pngmergealpha(tf, self.tsize) for tf in tfiles]

        self.heart = self.pngmergealpha("resources/heart.png", 50)
        self.heartmask = np.zeros_like(self.heart)
        self.heartmask[self.heart > 0] = 1

        self.balloonimg, self.balloonmask = self.pngmergealpha("resources/s/balloon.png", self.ssize, True)
        self.lockimg, self.lockmask = self.pngmergealpha("resources/s/lock.png", self.ssize, True)
        self.calculate()

    def getheart(self):
        return self.heart, self.heartmask

    def gethw(self):
        return self.h, self.w

    def gettimgs(self):
        return self.timgs[random.randint(0, len(self.timgs)-1)]

    def getdimgs(self):
        return self.dimgs[random.randint(0, len(self.dimgs)-1)]

    def getballoonimg(self):
        return self.balloonimg

    def getballoonmask(self):
        return self.balloonmask

    def getlockimg(self):
        return self.lockimg

    def getlockmask(self):
        return self.lockmask

    def gettmusic(self):
        self.tmusic = ["t1.mp3","t2.mp3","t3.mp3","t4.mp3"]
        return self.tmusic[random.randint(0, len(self.tmusic)-1)]

    def getballoonmusic(self):
        return "balloon.mp3"

    def getlockmusic(self):
        return "lock.mp3"

    def getopmusic(self):
        return "op.mp3"

    def getop(self):
        return "resources/op.mp4"

    def getwinmusic(self):
        return "win.mp3"

    def getwin(self):
        return "resources/win.mp4"

    def getlosemusic(self):
        return "lose.mp3"

    def getlose(self):
        return "resources/lose.mp4"

    def resize(self, img, size):
        h, w = img.shape[:2]
        ml = max(h, w)
        t = ml / size
        return cv2.resize(img, (int(w / t), int(h / t)))

    def pngmergealpha(self, imgpath, dsize, needmask = False):
        img = cv2.imread(imgpath, cv2.IMREAD_UNCHANGED)
        rimg = self.resize(img, dsize)
        mask = rimg[:,:,3]
        mask[mask > 0] = 1
        rimg = rimg[:,:,:-1]
        if needmask:
            mask3 = np.repeat(mask[:,:,np.newaxis], 3, 2)
            return rimg * mask3, mask
        else:
            return rimg * np.repeat(mask[:,:,np.newaxis], 3, 2)

    def calculate(self):
        self.h = 0
        self.w = 0
        tlist = self.timgs + self.dimgs
        tlist.append(self.balloonimg)
        tlist.append(self.lockimg)
        for img in tlist:
            th, tw = img.shape[:2]
            if th > self.h:
                self.h = th
            if tw > self.w:
                self.w = tw

    def loadMap(self, H, W):
        mapp = cv2.imread("resources/map.png")
        return cv2.resize(mapp, (W, H))


def cv2ImgAddText(img, text, left, top, textColor=(255, 0, 0), textSize=50):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    draw = ImageDraw.Draw(img)

    fontStyle = ImageFont.truetype(
        "font/simsun.ttc", textSize, encoding="utf-8")

    draw.text((left+1, top+1), text, (0, 0, 0), font=fontStyle)
    draw.text((left, top), text, textColor, font=fontStyle)

    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

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

    def __init__(self, x, y, speed_x, speed_y, img, skill, mask=None):
        self.x = x
        self.y = y
        self.speed_x = speed_x
        self.speed_y = speed_y
        self.img = img
        if mask is None:
            self.mask = np.zeros_like(img)
            self.mask[img > 0] = 1
        else:
            self.mask = np.repeat(mask[:,:,np.newaxis], 3, 2)
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
            if isinstance(self.skill, Balloon):
                _thread.start_new_thread(sound.thread_playsound, ("sound1",RES.getballoonmusic()))
            elif isinstance(self.skill, Lock):
                _thread.start_new_thread(sound.thread_playsound, ("sound1",RES.getlockmusic()))
            else:
                _thread.start_new_thread(sound.thread_playsound, ("sound1",RES.gettmusic()))
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

def PlayVideo(video_path, music, H, W):
    video = cv2.VideoCapture(video_path)
    _thread.start_new_thread(sound.thread_playsound, ("sound1", music))
    lastframe = None
    while True:
        grabbed, frame = video.read()
        if not grabbed:
            break
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break
        frame = cv2.resize(frame,(W, H))
        cv2.imshow("Game", frame)
        lastframe = frame
    video.release()
    return lastframe

GM = GameManager(1)
RES = Resources(120, 50, 60)


def create_ball():
    global GM
    global level
    h,w = RES.gethw()
    
    speed = level if level > 5 else 5

    speed_x = 0
    speed_y = 0
    while speed_x == 0 or speed_y == 0:
        speed_x = random.randint(-speed, speed)
        speed_y = random.randint(-speed, speed)

    x, y = randomXY(h+20, w+20)
    if inseg(x,y,w/2,h/2):
        x, y = randomXY(h+5, w+5)

    ratio = random.randint(1, 100)
    if ratio < 13:
        ratio = random.randint(1, 100)
        if ratio < 30: 
            b = Ball(x, y, speed_x, speed_y, RES.getballoonimg(), Balloon(0, GM), RES.getballoonmask())
        else:
            
            b = Ball(x, y, speed_x, speed_y, RES.getlockimg(), Lock(random.randint(1,10), GM), RES.getlockmask())
    else:
        b = Ball(x, y, speed_x, speed_y, RES.gettimgs(), Tansir(0, GM))
    balls.append(b) 

def ball_manager():
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
                    create_ball()
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

def drawUI(canvas, time, heart, mask):
    # draw HP
    startx = 0
    canvas = cv2ImgAddText(canvas, "HP: ", startx, 10)
    startx += 75
    # draw heart
    h, w = heart.shape[:2]
    if GM.lives == 0:
        canvas = cv2ImgAddText(canvas, "0 ", startx, 10)
    else:
        for l in range(GM.lives):
            canvas[10:10+h,startx:startx+w] = canvas[10:10+h,startx:startx+w] * (1-mask) + mask * heart
            startx += (w + 5)
    # draw time icon:
    startx += 15
    canvas = cv2ImgAddText(canvas, "Time: %.2f" %time, startx, 10)
    return canvas

def main(args):
    global showimg
    global H
    global W
    global currentTime
    global startTime
    global balls
    global currentIndex
    global checkimg
    global GM
    global level

    level = args.level
    restart = True
    du = detUtils()
    # videostream = "test.mp4"
    videostream = 0
    cap = cv2.VideoCapture(videostream)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if max(H, W) < 1000:
        H, W = H * scale, W * scale
    showimgt = RES.loadMap(H, W)
    heartimg, heartmask = RES.getheart()
    ret, frame = cap.read()
    print(RES.getop())
    print(RES.getopmusic())
    PlayVideo(RES.getop(), RES.getopmusic(), H, W)
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
        tdimg = RES.getdimgs()
        while True:
            
            ret, frame = cap.read()
            if videostream == 0:
                frame = cv2.flip(frame, 1)

            cv2.imshow("self", frame)

            if ret == True:

                frame = cv2.resize(frame, (W, H))
                currentTime = math.floor(time.time() - startTime)
                
                if currentTime % 4 == 0 and currentTime != llt:
                    tdimg = RES.getdimgs()
                    llt = currentTime
                dimg = copy.deepcopy(tdimg)
                
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
                    showimg = showimg.astype(np.uint8)
                    checkimg[t:b,l:r] = checkimg[t:b,l:r] + mask
                    if r < 0 or b < 0 or t >= H or l >= W:
                        gameover = True
                    else:
                        gameover = False
                    if not gameover:
                        gameover = ball_manager()
                    showimg = showimg.astype(np.uint8)
                    
                    # showimg = cv2.putText(showimg, "Lives: " + heart + "Pixel: %d"% int((right - left) * (bottom - top))+" Time: %.2f"% (time.time() - startTime), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 3)
                    # showimg = cv2ImgAddText(showimg, "Lives: %d" %GM.lives + "Pixel: %d"% int((right - left) * (bottom - top))+" Time: %.2f"% (time.time() - startTime), 0, 10)
                    showimg = drawUI(showimg, 100 - (time.time() - startTime), heartimg, heartmask)
                    if gameover:
                        # showimg = cv2.putText(showimg, "You Lose! Time: %2f" % (time.time() - startTime), (0, int(H/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
                        loseimg = PlayVideo(RES.getlose(), RES.getlosemusic(), H, W)
                        
                        loseimg = cv2ImgAddText(loseimg, "You Lose! Time: %.2f" % (100 - (time.time() - startTime)), int(W/5), int(H/2))
                        cv2.imshow('Game', loseimg)
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
                        showimg = showimg = cv2ImgAddText(showimg, "Keep your face in camera", 0, int(H/2))
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
                        showimg = showimg.astype(np.uint8)
                        checkimg[t:b,l:r] = checkimg[t:b,l:r] + mask
                        
                        gameover = ball_manager()
                        showimg = showimg.astype(np.uint8)
                        
                        
                        # showimg = cv2.putText(showimg, "Lives: " + heart + "Pixel: %d"% int((right - left) * (bottom - top))+" Time: %.2f"% (time.time() - startTime), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 3)
                        # showimg = cv2ImgAddText(showimg, "Lives: %d" %GM.lives + "Pixel: %d"% int((right - left) * (bottom - top))+" Time: %.2f"% (time.time() - startTime), 0, 10)
                        drawUI(showimg, 100 - (time.time() - startTime), heartimg, heartmask)
                        if gameover:
                            #showimg = cv2.putText(showimg, "You Lose! Time: %2f" % (time.time() - startTime), (0, int(H/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
                            
                            loseimg = PlayVideo(RES.getlose(), RES.getlosemusic(), H, W)
                            loseimg = cv2ImgAddText(loseimg, "You Lose! Time: %.2f" % (100 - (time.time() - startTime)), int(W/5), int(H/2))
                            cv2.imshow('Game', loseimg)
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

            if time.time() - startTime > 100:
                PlayVideo(RES.getwin(), RES.getwinmusic(), H, W)
                if cv2.waitKey(0) == ord('r'):
                    restart = True
                break

    cv2.destroyAllWindows()
    cap.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, default=5)
    args = parser.parse_args()
    main(args)