# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import cv2
import shutil
import argparse
import torch
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
from PIL import Image
from PIL import ImageDraw
import numpy as np
import time
#from scipy import ndimage
import math
#np.set_printoptions(threshold=np.inf)
from datetime import datetime as dt
import zipfile
import io
import matplotlib.pyplot as plt

#LOG_MODEを1にするとM0,M1のどれを実行しても、他２つのMでの推定値も記録しておく※M:Mehod
LOG_MODE = 0
# 大域領域(５００×５００画像を１２５×１２５に圧縮)で探索する場合は１
GLOBAL_OR_LOCAL = 1
# 最終的は把持位置を[Graspability]と[1-ひっかかり予測値]の調和平均で決定するなら１、閾値判定の場合は０に設定し、適切な閾値を該当箇所(1445行目あたり)で設定
Harmonic_OR_Threshold = 1

#最終的に求まった"TOP_NUMBER"個の把持位置情報を格納
#"Top_○○"は同じインデックスが配列名の値を表す。
Top_Graspability = np.array([])
Top_ImgX = np.array([])
Top_ImgY = np.array([])
Top_ImgZ = np.array([])#0〜255の輝度値
Top_Angle = np.array([])
Top_CountA = np.array([])
Top_CountB = np.array([])

#M1の最終把持位置のGraspabilityを格納する変数
_Graspability_M1 = 0

class CNN1(nn.Module):
    def __init__(self):
        super(CNN1, self).__init__()
        # the size of the inputs to each layer will be inferred
        self.conv1a=nn.Conv2d(1,32,16)
        self.conv2a=nn.Conv2d(32,64,8)
        self.conv1b=nn.Conv2d(1,32,16)
        self.conv2b=nn.Conv2d(32,64,8)
        self.conv3=nn.Conv2d(64,64,5)
        self.conv4=nn.Conv2d(64,64,3)
        self.l1=nn.Linear(4*4*64,1024)
        self.l2=nn.Linear(1024,1024)
        self.l3=nn.Linear(1024,2)
        self.dropout = nn.Dropout(0.5)

    def fwd(self,xa,xb,train):
        h = F.max_pool2d(F.relu(self.conv1a(xa)), 2)

        ha = F.max_pool2d(F.relu(self.conv2a(h)), 2)

        h = F.max_pool2d(F.relu(self.conv1b(xb)), 2)
        hb = F.max_pool2d(F.relu(self.conv2b(h)), 2)

        h = ha + hb

        h = F.max_pool2d(F.relu(self.conv3(h)), 2)
        h = F.max_pool2d(F.relu(self.conv4(h)), 2)

        t = h.view(1, -1)
        # h = F.dropout(F.relu(self.l1(h)),ratio=0.5, train=train) # chainer v1
        t = self.dropout(F.relu(self.l1(t))) # chainer v2

        # h = F.dropout(F.relu(self.l2(h)),ratio=0.5, train=train) # chain v1
        t = self.dropout(F.relu(self.l2(t))) # chain v2
        return self.l3(t)

    def output(self, xa_data, xb_data ,train=False):
        xa=tensor(xa_data.astype(xp.float32))
        xb=tensor(xb_data.astype(xp.float32))

        return F.softmax(self.fwd(xa,xb,train)).data

#Method1の関数
# ①５００×５００　→近傍切取り→　②２５０×２５０　→圧縮→　③１２５×１２５
def M1_LocalCut_and_Compress(METHOD,tstr,img_pass,x,y,count):
    img_depth = cv2.imread('{}'.format(img_pass),0)  #グレーで読み込み
    local_depth = img_depth[y : y + WINDOW_HEIGHT , x : x + WINDOW_WIDTH ]  #ラスタウィンドウ部分のみ切り抜き(WINDOW_HEIGHT × WINDOW_WIDTH)
    size = (OUTPUT_HEIGHT , OUTPUT_WIDTH)
    compressed_img = cv2.resize(local_depth,size)
    #cv2.imwrite('./Method{}/EXPERIMENT/{}/QUARTER_DEPTH/img{}_{}.bmp'.format(METHOD,tstr,count,count+ANGLE_NUMBER-1),compressed_img)    #これがchainerへの入力img_aとなる
    cv2.imwrite('./Method{}/EXPERIMENT/{}/QUARTER_DEPTH/img{}_{}.png'.format(METHOD,tstr,count,count+ANGLE_NUMBER-1),compressed_img)    #これがchainerへの入力img_aとなる

# ①２５０×２５０黒色　→ハンド描画→　②２５０×２５０　→圧縮→　③１２５×１２５
def M1_DrawHand_and_Compress(METHOD,tstr,angle,count):
    img = cv2.imread('./black250.png',0)  #グレーで読み込み
    finger_top_x = int( ( 250 / 2 ) + ( ( HAND_WIDTH / 2 ) * BEFORE_TO_AFTER * math.cos(angle) ) )
    finger_top_y = int( ( 250 / 2 ) - ( ( HAND_WIDTH / 2 ) * BEFORE_TO_AFTER * math.sin(angle) ) )
    finger_bottom_x = int( ( 250 / 2 ) - ( ( HAND_WIDTH / 2 ) * BEFORE_TO_AFTER * math.cos(angle) ) )
    finger_bottom_y = int( ( 250 / 2 ) + ( ( HAND_WIDTH / 2 ) * BEFORE_TO_AFTER * math.sin(angle) ) )
    # ハンドを白色描画
    cv2.circle(img, (finger_top_x, finger_top_y), 10, (255, 255, 255), -1)
    cv2.circle(img, (finger_bottom_x, finger_bottom_y), 10, (255, 255, 255), -1)
    cv2.line(img, (finger_top_x, finger_top_y), (finger_bottom_x, finger_bottom_y), (255, 255, 255), 5)
    size = (OUTPUT_HEIGHT , OUTPUT_WIDTH)   #chainerへの入力となるサイズ
    compressed_img = cv2.resize(img,size)
    cv2.imwrite('./Method{}/EXPERIMENT/{}/QUARTER_HAND/img{}.png'.format(METHOD,tstr,count),compressed_img)

def M1_DrawHandWithWork_and_Compress(METHOD,tstr,img_pass,center_x,center_y,angle,count):
    img = cv2.imread('{}'.format(img_pass),0)  #グレーで読み込み
    finger_top_x = int( ( center_x ) + ( ( HAND_WIDTH / 2 ) * BEFORE_TO_AFTER * math.cos(angle) ) )
    finger_top_y = int( ( center_y ) - ( ( HAND_WIDTH / 2 ) * BEFORE_TO_AFTER * math.sin(angle) ) )
    finger_bottom_x = int( ( center_x ) - ( ( HAND_WIDTH / 2 ) * BEFORE_TO_AFTER * math.cos(angle) ) )
    finger_bottom_y = int( ( center_y ) + ( ( HAND_WIDTH / 2 ) * BEFORE_TO_AFTER * math.sin(angle) ) )
    # ハンドを白色描画
    cv2.circle(img, (finger_top_x, finger_top_y), 10, (255, 255, 255), -1)
    cv2.circle(img, (finger_bottom_x, finger_bottom_y), 10, (255, 255, 255), -1)
    cv2.line(img, (finger_top_x, finger_top_y), (finger_bottom_x, finger_bottom_y), (255, 255, 255), 5)
    #cv2.imwrite('./Method{}/EXPERIMENT/{}/HAND_WITH_WORK/img{}.bmp'.format(METHOD,tstr,count),img)
    cv2.imwrite('./Method{}/EXPERIMENT/{}/HAND_WITH_WORK/img{}.png'.format(METHOD,tstr,count),img)
    #x座標のはみ出し判定
    if center_x - (WINDOW_WIDTH // 2) <= 0:
        lower_x = 0
        upper_x = WINDOW_WIDTH
    elif TRIMMED_WIDTH <= center_x + (WINDOW_WIDTH // 2) :
        lower_x = TRIMMED_WIDTH - WINDOW_WIDTH
        upper_x = TRIMMED_WIDTH
    else:
        lower_x = center_x - (WINDOW_WIDTH // 2)
        upper_x = center_x + (WINDOW_WIDTH // 2)
    #y座標のはみ出し判定
    if center_y - (WINDOW_HEIGHT // 2) <= 0:
        lower_y = 0
        upper_y = WINDOW_WIDTH
    elif TRIMMED_HEIGHT <= center_y + (WINDOW_HEIGHT // 2) :
        lower_y = TRIMMED_HEIGHT - WINDOW_HEIGHT
        upper_y = TRIMMED_HEIGHT
    else:
        lower_y = center_y - (WINDOW_HEIGHT // 2)
        upper_y = center_y + (WINDOW_HEIGHT // 2)
    #500×500→250×250をそのまま切り取った後、圧縮
    local_hand_with_work = img[int(lower_y):int(upper_y),int(lower_x):int(upper_x)]
    size = (OUTPUT_HEIGHT , OUTPUT_WIDTH)#chainerへの入力となるサイズ
    compressed_hand_with_work = cv2.resize(local_hand_with_work,size)
    cv2.imwrite('./Method{}/EXPERIMENT/{}/QUARTER_HAND_WITH_WORK/img{}.png'.format(METHOD,tstr,count),compressed_hand_with_work)

def M1_DrawBestHandWithWork(img_pass,center_x,center_y,angle,B,G,R):
    img = cv2.imread('{}'.format(img_pass),1)  #※グレーで読み込み（0）すると，円や直線も白か黒しか描けなくなるので注意

    finger_top_x = int( ( center_x ) + ( ( HAND_WIDTH / 2 ) * BEFORE_TO_AFTER * math.cos(angle) ) )
    finger_top_y = int( ( center_y ) - ( ( HAND_WIDTH / 2 ) * BEFORE_TO_AFTER * math.sin(angle) ) )
    finger_bottom_x = int( ( center_x ) - ( ( HAND_WIDTH / 2 ) * BEFORE_TO_AFTER * math.cos(angle) ) )
    finger_bottom_y = int( ( center_y ) + ( ( HAND_WIDTH / 2 ) * BEFORE_TO_AFTER * math.sin(angle) ) )

    cv2.circle(img, (finger_top_x, finger_top_y), 10, (B, G, R), -1)
    cv2.circle(img, (finger_bottom_x, finger_bottom_y), 10, (B, G, R), -1)
    cv2.line(img, (finger_top_x, finger_top_y), (finger_bottom_x, finger_bottom_y), (B, G, R), 5)

    cv2.imwrite('{}'.format(img_pass),img)

def M1_Graspability(METHOD,tstr,img_pass,final_center_x,final_center_y,final_center_z,final_angle):

    #=== 準備（ハンドのテンプレートを生成）=================================================================
    #else:
    #グレースケールで読み込み
    Ht_original = cv2.imread("./black250.png",0)
    Hc_original = cv2.imread("./black250.png",0)
    TEMPLATE_SIZE = 250
    HAND_THICKNESS_X = 15#シミュレーション時は2だが、実際のハンドに合わせて15mm
    HAND_THICKNESS_Y = 5#シミュレーション時は5だが、実際のハンドに合わせて25mm
    BEFORE_TO_AFTER = 500/225

    #       L1  L2              R1  R2
    #       L4  L3              R4  R3

    L1x = int ( ( TEMPLATE_SIZE / 2 ) - ( ( ( HAND_WIDTH / 2 ) + HAND_THICKNESS_X ) * BEFORE_TO_AFTER ) )
    L2x = int ( ( TEMPLATE_SIZE / 2 ) - ( ( ( HAND_WIDTH / 2 ) +        0       ) * BEFORE_TO_AFTER ) )
    L3x = int ( ( TEMPLATE_SIZE / 2 ) - ( ( ( HAND_WIDTH / 2 ) +        0       ) * BEFORE_TO_AFTER ) )
    L4x = int ( ( TEMPLATE_SIZE / 2 ) - ( ( ( HAND_WIDTH / 2 ) + HAND_THICKNESS_X ) * BEFORE_TO_AFTER ) )
    R1x = int ( ( TEMPLATE_SIZE / 2 ) + ( ( ( HAND_WIDTH / 2 ) +        0       ) * BEFORE_TO_AFTER ) )
    R2x = int ( ( TEMPLATE_SIZE / 2 ) + ( ( ( HAND_WIDTH / 2 ) + HAND_THICKNESS_X ) * BEFORE_TO_AFTER ) )
    R3x = int ( ( TEMPLATE_SIZE / 2 ) + ( ( ( HAND_WIDTH / 2 ) + HAND_THICKNESS_X ) * BEFORE_TO_AFTER ) )
    R4x = int ( ( TEMPLATE_SIZE / 2 ) + ( ( ( HAND_WIDTH / 2 ) +        0       ) * BEFORE_TO_AFTER ) )

    L1y = int ( ( TEMPLATE_SIZE / 2 ) - ( (  HAND_THICKNESS_Y / 2 ) * BEFORE_TO_AFTER ) )
    L2y = int ( ( TEMPLATE_SIZE / 2 ) - ( (  HAND_THICKNESS_Y / 2 ) * BEFORE_TO_AFTER ) )
    L3y = int ( ( TEMPLATE_SIZE / 2 ) + ( (  HAND_THICKNESS_Y / 2 ) * BEFORE_TO_AFTER ) )
    L4y = int ( ( TEMPLATE_SIZE / 2 ) + ( (  HAND_THICKNESS_Y / 2 ) * BEFORE_TO_AFTER ) )
    R1y = int ( ( TEMPLATE_SIZE / 2 ) - ( (  HAND_THICKNESS_Y / 2 ) * BEFORE_TO_AFTER ) )
    R2y = int ( ( TEMPLATE_SIZE / 2 ) - ( (  HAND_THICKNESS_Y / 2 ) * BEFORE_TO_AFTER ) )
    R3y = int ( ( TEMPLATE_SIZE / 2 ) + ( (  HAND_THICKNESS_Y / 2 ) * BEFORE_TO_AFTER ) )
    R4y = int ( ( TEMPLATE_SIZE / 2 ) + ( (  HAND_THICKNESS_Y / 2 ) * BEFORE_TO_AFTER ) )

    #左上の角と右下の角を指定すると四角形が描かれる
    cv2.rectangle(Hc_original, (L1x, L1y), (L3x,L3y),(255, 255, 255), -1)
    cv2.rectangle(Hc_original, (R1x, R1y), (R3x,R3y),(255, 255, 255), -1)
    cv2.rectangle(Ht_original, (L2x, L2y), (R4x,R4y),(255, 255, 255), -1)

    #os.makedirs("./tmp")
    cv2.imwrite('./Method{}/EXPERIMENT/{}/M0tmp/Ht_original.png'.format(METHOD,tstr),Ht_original)
    cv2.imwrite('./Method{}/EXPERIMENT/{}/M0tmp/Hc_original.png'.format(METHOD,tstr),Hc_original)
    # ここまで準備 ====================================================================================================
    GripperD = 50
    GaussianKernelSize = 75
    GaussianSigma = 25

    Depth_original = cv2.imread("{}".format(img_pass),0)
    _, Wt = cv2.threshold(Depth_original,final_center_z+GripperD/2,255,cv2.THRESH_BINARY)
    _, Wc = cv2.threshold(Depth_original,final_center_z-GripperD/2,255,cv2.THRESH_BINARY)
    cv2.imwrite("./Method{}/EXPERIMENT/{}/M0tmp/Wt.png".format(METHOD,tstr),Wt)
    cv2.imwrite("./Method{}/EXPERIMENT/{}/M0tmp/Wc.png".format(METHOD,tstr),Wc)

    Ht = Image.open("./Method{}/EXPERIMENT/{}/M0tmp/Ht_original.png".format(METHOD,tstr))
    Ht = Ht.rotate(final_angle*180/math.pi)
    Ht.save("./Method{}/EXPERIMENT/{}/M0tmp/Ht.png".format(METHOD,tstr))
    Hc = Image.open("./Method{}/EXPERIMENT/{}/M0tmp/Hc_original.png".format(METHOD,tstr))
    Hc = Hc.rotate(final_angle*180/math.pi)
    Hc.save("./Method{}/EXPERIMENT/{}/M0tmp/Hc.png".format(METHOD,tstr))
    #ハンドモデルの接触領域計算
    Ht = np.array(Image.open("./Method{}/EXPERIMENT/{}/M0tmp/Ht.png".format(METHOD,tstr)).convert('L'))
    T = cv2.filter2D(Wt,-1,Ht)
    cv2.imwrite("./Method{}/EXPERIMENT/{}/M0tmp/T.png".format(METHOD,tstr),T)
    #ハンドモデルの衝突領域計算
    Hc = np.array(Image.open("./Method{}/EXPERIMENT/{}/M0tmp/Hc.png".format(METHOD,tstr)).convert('L'))
    C = cv2.filter2D(Wc,-1,Hc)
    cv2.imwrite("./Method{}/EXPERIMENT/{}/M0tmp/C.png".format(METHOD,tstr),C)
    #ハンドモデルの衝突領域を反転（補集合）
    Cbar = 255 - C
    cv2.imwrite("./Method{}/EXPERIMENT/{}/M0tmp/Cbar.png".format(METHOD,tstr),Cbar)
    #Graspabilityマップを作成（ぼかし前）
    T_and_Cbar = T & Cbar
    cv2.imwrite("./Method{}/EXPERIMENT/{}/M0tmp/T_and_Cbar.png".format(METHOD,tstr),T_and_Cbar)
    #Graspabilityマップを完成（ぼかし後）
    global _Graspability_M1
    G = cv2.GaussianBlur(T_and_Cbar,(GaussianKernelSize,GaussianKernelSize),GaussianSigma,GaussianSigma)
    cv2.imwrite("./Method{}/EXPERIMENT/{}/M0tmp/G.png".format(METHOD,tstr),G)
    _Graspability_M1 = G[int(final_center_y)][int(final_center_x)]

def Graspability(METHOD,tstr,img_pass):

    #モーションファイルがあれば２回目以降と判断し、テンプレートの生成はスキップ
    #=== 2回目以降の処理
    # if(os.path.exists("./MOTION/motion.txt")):
    #     print("REMOVED MOTIONFILE")
    #     os.remove("./MOTION/motion.txt")
    #=== 準備（ハンドのテンプレートを生成）=================================================================
    #else:
    #グレースケールで読み込み
    Ht_original = cv2.imread("./black250.png",0)
    Hc_original = cv2.imread("./black250.png",0)
    TEMPLATE_SIZE = 250
    HAND_THICKNESS_X = 15#シミュレーション時は2だが、実際のハンドに合わせて15mm
    HAND_THICKNESS_Y = 5#シミュレーション時は5だが、実際のハンドに合わせて25mm
    BEFORE_TO_AFTER = 500/225

    #       L1  L2              R1  R2
    #       L4  L3              R4  R3

    L1x = int ( ( TEMPLATE_SIZE / 2 ) - ( ( ( HAND_WIDTH / 2 ) + HAND_THICKNESS_X ) * BEFORE_TO_AFTER ) )
    L2x = int ( ( TEMPLATE_SIZE / 2 ) - ( ( ( HAND_WIDTH / 2 ) +        0       ) * BEFORE_TO_AFTER ) )
    L3x = int ( ( TEMPLATE_SIZE / 2 ) - ( ( ( HAND_WIDTH / 2 ) +        0       ) * BEFORE_TO_AFTER ) )
    L4x = int ( ( TEMPLATE_SIZE / 2 ) - ( ( ( HAND_WIDTH / 2 ) + HAND_THICKNESS_X ) * BEFORE_TO_AFTER ) )
    R1x = int ( ( TEMPLATE_SIZE / 2 ) + ( ( ( HAND_WIDTH / 2 ) +        0       ) * BEFORE_TO_AFTER ) )
    R2x = int ( ( TEMPLATE_SIZE / 2 ) + ( ( ( HAND_WIDTH / 2 ) + HAND_THICKNESS_X ) * BEFORE_TO_AFTER ) )
    R3x = int ( ( TEMPLATE_SIZE / 2 ) + ( ( ( HAND_WIDTH / 2 ) + HAND_THICKNESS_X ) * BEFORE_TO_AFTER ) )
    R4x = int ( ( TEMPLATE_SIZE / 2 ) + ( ( ( HAND_WIDTH / 2 ) +        0       ) * BEFORE_TO_AFTER ) )

    L1y = int ( ( TEMPLATE_SIZE / 2 ) - ( (  HAND_THICKNESS_Y / 2 ) * BEFORE_TO_AFTER ) )
    L2y = int ( ( TEMPLATE_SIZE / 2 ) - ( (  HAND_THICKNESS_Y / 2 ) * BEFORE_TO_AFTER ) )
    L3y = int ( ( TEMPLATE_SIZE / 2 ) + ( (  HAND_THICKNESS_Y / 2 ) * BEFORE_TO_AFTER ) )
    L4y = int ( ( TEMPLATE_SIZE / 2 ) + ( (  HAND_THICKNESS_Y / 2 ) * BEFORE_TO_AFTER ) )
    R1y = int ( ( TEMPLATE_SIZE / 2 ) - ( (  HAND_THICKNESS_Y / 2 ) * BEFORE_TO_AFTER ) )
    R2y = int ( ( TEMPLATE_SIZE / 2 ) - ( (  HAND_THICKNESS_Y / 2 ) * BEFORE_TO_AFTER ) )
    R3y = int ( ( TEMPLATE_SIZE / 2 ) + ( (  HAND_THICKNESS_Y / 2 ) * BEFORE_TO_AFTER ) )
    R4y = int ( ( TEMPLATE_SIZE / 2 ) + ( (  HAND_THICKNESS_Y / 2 ) * BEFORE_TO_AFTER ) )

    #左上の角と右下の角を指定すると四角形が描かれる
    cv2.rectangle(Hc_original, (L1x, L1y), (L3x,L3y),(255, 255, 255), -1)
    cv2.rectangle(Hc_original, (R1x, R1y), (R3x,R3y),(255, 255, 255), -1)
    cv2.rectangle(Ht_original, (L2x, L2y), (R4x,R4y),(255, 255, 255), -1)

    #os.makedirs("./tmp")
    cv2.imwrite('./Method{}/EXPERIMENT/{}/tmp/Ht_original.png'.format(METHOD,tstr),Ht_original)
    cv2.imwrite('./Method{}/EXPERIMENT/{}/tmp/Hc_original.png'.format(METHOD,tstr),Hc_original)
    # ここまで準備 ====================================================================================================
    HandRotationStep = 22.5
    HandDepthStep = 50 #25でも可能
    GripperD = 50
    InitialHandDepth = 0
    FinalHandDepth = 201
    GaussianKernelSize = 75
    GaussianSigma = 25
    CountA = 0
    CountB = 0
    #"all_○○"配列を初期化
    #all_LabelCenter_Area = np.array([])
    all_LabelCenter_X = np.array([])
    all_LabelCenter_Y = np.array([])
    all_LabelCenter_CountA = np.array([])
    all_LabelCenter_CountB = np.array([])
    all_LabelCenter_Graspability = np.array([])

    #深さパターン数だけマップ生成
    for HandDepth in range(InitialHandDepth,FinalHandDepth,HandDepthStep):
        CountA += 1
        #Depth_original = cv2.imread("./DEPTH_TMP/depth.bmp",0)
        Depth_original = cv2.imread("{}".format(img_pass),0)
        _, Wt = cv2.threshold(Depth_original,HandDepth+GripperD,255,cv2.THRESH_BINARY)
        _, Wc = cv2.threshold(Depth_original,HandDepth,255,cv2.THRESH_BINARY)

        cv2.imwrite("./Method{}/EXPERIMENT/{}/tmp/Wt{}.png".format(METHOD,tstr,CountA),Wt)
        cv2.imwrite("./Method{}/EXPERIMENT/{}/tmp/Wc{}.png".format(METHOD,tstr,CountA),Wc)

        CountB = 0
        #回転パターン数だけマップ生成
        for HandRotation in np.arange(0,180,HandRotationStep):
            CountB += 1
            if(CountA == 1):
                Ht = Image.open("./Method{}/EXPERIMENT/{}/tmp/Ht_original.png".format(METHOD,tstr))
                Ht = Ht.rotate(HandRotation)
                Ht.save("./Method{}/EXPERIMENT/{}/tmp/Ht{}.png".format(METHOD,tstr,CountB))
                Hc = Image.open("./Method{}/EXPERIMENT/{}/tmp/Hc_original.png".format(METHOD,tstr))
                Hc = Hc.rotate(HandRotation)
                Hc.save("./Method{}/EXPERIMENT/{}/tmp/Hc{}.png".format(METHOD,tstr,CountB))
            '''
            else:
                Ht = Image.open("./BEST/{}/tmp/Ht{}.png".format(CountB))
                Hc = Image.open("./BEST/{}/tmp/Hc{}.png".format(CountB))
            '''

            #Wt = cv2.imread("./tmp/Wt{}.png".format(CountA),0)
            #Wc = cv2.imread("./tmp/Wc{}.png".format(CountA),0)

            #ハンドモデルの接触領域計算
            Ht = np.array(Image.open("./Method{}/EXPERIMENT/{}/tmp/Ht{}.png".format(METHOD,tstr,CountB)).convert('L'))
            T = cv2.filter2D(Wt,-1,Ht)
            cv2.imwrite("./Method{}/EXPERIMENT/{}/tmp/T{}_{}.png".format(METHOD,tstr,CountA,CountB),T)

            #ハンドモデルの衝突領域計算
            Hc = np.array(Image.open("./Method{}/EXPERIMENT/{}/tmp/Hc{}.png".format(METHOD,tstr,CountB)).convert('L'))
            C = cv2.filter2D(Wc,-1,Hc)
            cv2.imwrite("./Method{}/EXPERIMENT/{}/tmp/C{}_{}.png".format(METHOD,tstr,CountA,CountB),C)

            #ハンドモデルの衝突領域を反転（補集合）
            Cbar = 255 - C
            cv2.imwrite("./Method{}/EXPERIMENT/{}/tmp/Cbar{}_{}.png".format(METHOD,tstr,CountA,CountB),Cbar)

            #Graspabilityマップを作成（ぼかし前）
            T_and_Cbar = T & Cbar
            cv2.imwrite("./Method{}/EXPERIMENT/{}/tmp/T_and_Cbar{}_{}.png".format(METHOD,tstr,CountA,CountB),T_and_Cbar)

            #Graspabilityマップを完成（ぼかし後）
            G = cv2.GaussianBlur(T_and_Cbar,(GaussianKernelSize,GaussianKernelSize),GaussianSigma,GaussianSigma)
            cv2.imwrite("./Method{}/EXPERIMENT/{}/tmp/G{}_{}.png".format(METHOD,tstr,CountA,CountB),G)


            src = cv2.imread("./Method{}/EXPERIMENT/{}/tmp/T_and_Cbar{}_{}.png".format(METHOD,tstr,CountA,CountB),0)
            #この画像は0か255の輝度値しか持たないが、安全のためしきい値を122に設定
            ret,thresh = cv2.threshold(src,122,255,cv2.THRESH_BINARY)
            T_and_Cbar_Labeled = cv2.connectedComponentsWithStats(thresh)

            #ラベリング =========================================================================
            src = cv2.imread("./Method{}/EXPERIMENT/{}/tmp/T_and_Cbar{}_{}.png".format(METHOD,tstr,CountA,CountB),0)
            #ラベリングするには２値化処理が必須。122を閾値として「0or255の画像」→「バイナリ画像」に変換
            ret,thresh = cv2.threshold(src,122,255,cv2.THRESH_BINARY)
            T_and_Cbar_Labeled = cv2.connectedComponentsWithStats(thresh)
            #必要な情報だけ抽出し、
            data = np.delete(T_and_Cbar_Labeled[2],0,0)#２次元配列data = [[ラベル１のバウンディングボックス左上X, ラベル１のバウンディングボックス左上Y, ラベル１のバウンディングボックスX幅, ラベル１のバウンディングボックスY幅, ラベル１の面積], ...]
            center = np.delete(T_and_Cbar_Labeled[3],0,0)#２次元配列center = [[ラベル１の中心x, ラベル１の中心y], ...]
            #本当に必要な情報だけさらに抽出
            #LabelCenter_Area = data[:,4]#1次元配列all_LabelCenter_Area = [ラベル１の面積, ラベル２の面積, ...]
            LabelCenter_X = center[:,0]#1次元配列all_LabelCenter_X = [ラベル１の中心X, ラベル２の中心X, ...]
            LabelCenter_Y = center[:,1]#1次元配列all_LabelCenter_Y = [ラベル１の中心Y, ラベル２の中心Y, ...]

            # print("ラベル数:",T_and_Cbar_Labeled[0]-1)

            #ラベル数分の"CountA"の値を格納
            _tmpA = np.zeros(LabelCenter_X.shape[0],dtype=int)
            _tmpA.fill(CountA)
            #print(_tmpA)
            #ラベル数分の"CountB"の値を格納
            _tmpB = np.zeros(LabelCenter_X.shape[0],dtype=int)
            _tmpB.fill(CountB)
            #print(_tmpB)

            #"all_○○"は同じインデックスが配列名の値を表す。
            #all_LabelCenter_Area = np.append(all_LabelCenter_Area,LabelCenter_Area)
            all_LabelCenter_X = np.append(all_LabelCenter_X,LabelCenter_X)
            all_LabelCenter_Y = np.append(all_LabelCenter_Y,LabelCenter_Y)
            all_LabelCenter_CountA = np.append(all_LabelCenter_CountA,_tmpA)
            all_LabelCenter_CountB = np.append(all_LabelCenter_CountB,_tmpB)
            #Graspabilityは個別に求める必要がある
            for i in range(LabelCenter_X.shape[0]):
                all_LabelCenter_Graspability = np.append(all_LabelCenter_Graspability,G[int(LabelCenter_Y[i])][int(LabelCenter_X[i])])
            #ここまでラベリング関連処理 ===================================================================
    #=============================　ここまでGraspabilityマップ生成ステップ　　==============================
    #=============================　ここからGraspability最大位置探索ステップ　==============================
    #Pythonの大域変数は"global"をつけないと操作できない
    global Top_Graspability
    global Top_ImgX
    global Top_ImgY
    global Top_ImgZ
    global Top_Angle
    global Top_CountA
    global Top_CountB

    sorted_index_of_Graspability = np.argsort(all_LabelCenter_Graspability)[::-1]
    # for i in range(all_LabelCenter_Area.shape[0]):
    #     #index = int( sorted_index_of_Area[_cnt] )
    #     #_Area = all_LabelCenter_Area[index]
    #     _x = int(all_LabelCenter_X[i])
    #     _y = int(all_LabelCenter_Y[i])
    #     _CountA = int(all_LabelCenter_CountA[i])
    #     _CountB = int(all_LabelCenter_CountB[i])
    #     _GraspabilityMap = cv2.imread("./Method{}/EXPERIMENT/{}/tmp/G{}_{}.png".format(METHOD,tstr,_CountA,_CountB),0)
    #     all_LabelCenter_Graspability = np.append(all_LabelCenter_Graspability,_GraspabilityMap[_y][_x])

    #把持位置が近いものはスキップ
    _threshold_distance = 50
    #print (all_LabelCenter_Area.shape[0])
    _cnt = 0 #_cntは候補がTOP_NUMBER個見つかるまで（top_countがTOP_NUMBERと一致するまで）カウントアップし続ける
    top_count = 0#これがTOP_NUMBERになるまで、または、Areaを持つ候補がなくなるまで続ける:
    while (top_count < TOP_NUMBER and _cnt < all_LabelCenter_Graspability.shape[0]):
        #第"top_count"番目に面積の大きいラベルについて
        index = int( sorted_index_of_Graspability[_cnt] )
        #_Area = all_LabelCenter_Graspability[index]
        _x = int(all_LabelCenter_X[index])
        _y = int(all_LabelCenter_Y[index])
        _CountA = int(all_LabelCenter_CountA[index])
        _CountB = int(all_LabelCenter_CountB[index])
        _Graspability = all_LabelCenter_Graspability[index]
        #_GraspabilityMap = cv2.imread("./Method{}/EXPERIMENT/{}/tmp/G{}_{}.png".format(METHOD,tstr,_CountA,_CountB),0)

        #端っこ過ぎる把持位置（上下左右から"DismissAreaWidth_Graspability"ピクセル以内）は除外
        if (DismissAreaWidth_Graspability < _x) and (_x < TRIMMED_WIDTH-DismissAreaWidth_Graspability) and (DismissAreaWidth_Graspability < _y ) and (_y < TRIMMED_HEIGHT-DismissAreaWidth_Graspability):
            if top_count == 0:
                ###
                Top_Graspability = np.append(Top_Graspability,_Graspability)
                Top_ImgX = np.append(Top_ImgX,_x)
                Top_ImgY = np.append(Top_ImgY,_y)
                Top_ImgZ = np.append(Top_ImgZ,int(InitialHandDepth + HandDepthStep*(_CountA-1) + GripperD/2))
                #回転角はラジアンで保存
                Top_Angle = np.append(Top_Angle,HandRotationStep*(_CountB-1)*(math.pi)/180)
                #ログ用にCountA,CountBもとっておく
                Top_CountA = np.append(Top_CountA,_CountA)
                Top_CountB = np.append(Top_CountB,_CountB)
                top_count += 1
            else :
                for i in range (top_count):
                    #既存上位の候補に、把持位置が近いものは候補外
                    if (_x - Top_ImgX[i])*(_x - Top_ImgX[i])+(_y - Top_ImgY[i])*(_y - Top_ImgY[i]) < _threshold_distance*_threshold_distance:
                        break
                    elif i == top_count - 1:
                        Top_Graspability = np.append(Top_Graspability,_Graspability)
                        Top_ImgX = np.append(Top_ImgX,_x)
                        Top_ImgY = np.append(Top_ImgY,_y)
                        Top_ImgZ = np.append(Top_ImgZ,int(InitialHandDepth + HandDepthStep*(_CountA-1) + GripperD/2))
                        #回転角はラジアンで保存
                        Top_Angle = np.append(Top_Angle,HandRotationStep*(_CountB-1)*(math.pi)/180)
                        #ログ用にCountA,CountBもとっておく
                        Top_CountA = np.append(Top_CountA,_CountA)
                        Top_CountB = np.append(Top_CountB,_CountB)
                        top_count += 1
        _cnt += 1
    #もしTOP_NUMBER個見つからなかった場合は距離を気にせずGraspabilityの大きい順のデータで残りを埋める
    if top_count < TOP_NUMBER:
        print("探索失敗（距離を無視してGraspabilityの大きい順で代用します）")
        #一旦リセット
        Top_Graspability = np.array([])
        Top_ImgX = np.array([])
        Top_ImgY = np.array([])
        Top_ImgZ = np.array([])#0〜255の輝度値
        Top_Angle = np.array([])
        Top_CountA = np.array([])
        Top_CountB = np.array([])
        i=0
        _detected=0
        print(TOP_NUMBER)
        while (_detected < TOP_NUMBER):
            index = int( sorted_index_of_Graspability[i] )
            #_Area = all_LabelCenter_Area[index]
            _x = int(all_LabelCenter_X[index])
            _y = int(all_LabelCenter_Y[index])

            if (DismissAreaWidth_Graspability < _x) and (_x < TRIMMED_WIDTH-DismissAreaWidth_Graspability) and (DismissAreaWidth_Graspability < _y ) and (_y < TRIMMED_HEIGHT-DismissAreaWidth_Graspability):
                _CountA = int(all_LabelCenter_CountA[index])
                _CountB = int(all_LabelCenter_CountB[index])
                _Graspability = int(all_LabelCenter_Graspability[index])
                #_GraspabilityMap = cv2.imread("./Method{}/EXPERIMENT/{}/tmp/G{}_{}.png".format(METHOD,tstr,_CountA,_CountB),0)
                Top_Graspability = np.append(Top_Graspability,_Graspability)
                Top_ImgX = np.append(Top_ImgX,_x)
                Top_ImgY = np.append(Top_ImgY,_y)
                Top_ImgZ = np.append(Top_ImgZ,int(InitialHandDepth + HandDepthStep*(_CountA-1) + GripperD/2))
                #回転角はラジアンで保存
                Top_Angle = np.append(Top_Angle,HandRotationStep*(_CountB-1)*(math.pi)/180)
                #ログ用にCountA,CountBもとっておく
                Top_CountA = np.append(Top_CountA,_CountA)
                Top_CountB = np.append(Top_CountB,_CountB)
                _detected+=1 #探索範囲内のやつ1個見つけた
            i += 1 #ループ続行
            print ("i={}".format(i))
#入力depth画像をchainerへの入力サイズへ変形
def M0_Draw_and_LocalCut_and_Compress(METHOD,tstr,img_pass):
    for top_count in range(TOP_NUMBER):
        ImgX = Top_ImgX[top_count]
        ImgY = Top_ImgY[top_count]
        ImgZ = Top_ImgZ[top_count]
        Angle = Top_Angle[top_count]
        #
        img_depth = cv2.imread('{}'.format(img_pass),0)  #グレーで読み込み
        #=================================== HAND画像 および HAND_WITH_WORK画像 ===============================================
        #①500×500の黒一様画像に描画→②近傍250×250を切り取り→③125×125に圧縮
        img_hand_with_work = cv2.imread('{}'.format(img_pass),0)  #グレーで読み込み
        img_hand = cv2.imread('./black500.png',0)  #グレーで読み込み
        #finger_topは画像上で上側の指で，finger_bottomは画像上で下側の指
        finger_top_x = int( ImgX + ( ( HAND_WIDTH / 2 ) * BEFORE_TO_AFTER * math.cos(Angle) ) )
        finger_top_y = int( ImgY - ( ( HAND_WIDTH / 2 ) * BEFORE_TO_AFTER * math.sin(Angle) ) )
        finger_bottom_x = int( ImgX - ( ( HAND_WIDTH / 2 ) * BEFORE_TO_AFTER * math.cos(Angle) ) )
        finger_bottom_y = int( ImgY + ( ( HAND_WIDTH / 2 ) * BEFORE_TO_AFTER * math.sin(Angle) ) )
        #ハンドを白で描画
        cv2.circle(img_hand, (finger_top_x, finger_top_y), 10, (255, 255, 255), -1)
        cv2.circle(img_hand, (finger_bottom_x, finger_bottom_y), 10, (255, 255, 255), -1)
        cv2.line(img_hand, (finger_top_x, finger_top_y), (finger_bottom_x, finger_bottom_y), (255, 255, 255), 5)
        #cv2.circle(img_hand, (finger_top_x, finger_top_y), 10, (int(ImgZ), int(ImgZ), int(ImgZ)), -1)
        #cv2.circle(img_hand, (finger_bottom_x, finger_bottom_y), 10, (int(ImgZ), int(ImgZ), int(ImgZ)), -1)
        #cv2.line(img_hand, (finger_top_x, finger_top_y), (finger_bottom_x, finger_bottom_y), (int(ImgZ), int(ImgZ), int(ImgZ)), 5)
        #HAND_WITH_WORKは見やすいように白色で描画
        cv2.circle(img_hand_with_work, (finger_top_x, finger_top_y), 10, (255, 255, 255), -1)
        cv2.circle(img_hand_with_work, (finger_bottom_x, finger_bottom_y), 10, (255, 255, 255), -1)
        cv2.line(img_hand_with_work, (finger_top_x, finger_top_y), (finger_bottom_x, finger_bottom_y), (255, 255, 255), 5)
        #x座標のはみ出し判定
        if ImgX - (WINDOW_WIDTH // 2) <= 0:
            lower_x = 0
            upper_x = WINDOW_WIDTH
        elif TRIMMED_WIDTH <= ImgX + (WINDOW_WIDTH // 2) :
            lower_x = TRIMMED_WIDTH - WINDOW_WIDTH
            upper_x = TRIMMED_WIDTH
        else:
            lower_x = ImgX - (WINDOW_WIDTH // 2)
            upper_x = ImgX + (WINDOW_WIDTH // 2)
        #y座標のはみ出し判定
        if ImgY - (WINDOW_HEIGHT // 2) <= 0:
            lower_y = 0
            upper_y = WINDOW_WIDTH
        elif TRIMMED_HEIGHT <= ImgY + (WINDOW_HEIGHT // 2) :
            lower_y = TRIMMED_HEIGHT - WINDOW_HEIGHT
            upper_y = TRIMMED_HEIGHT
        else:
            lower_y = ImgY - (WINDOW_HEIGHT // 2)
            upper_y = ImgY + (WINDOW_HEIGHT // 2)

        cv2.imwrite('./Method{}/EXPERIMENT/{}/HAND_WITH_WORK/img{}.png'.format(METHOD,tstr,top_count),img_hand_with_work)
        #500×500→250×250をそのまま切り取り
        local_depth = img_depth[int(lower_y):int(upper_y),int(lower_x):int(upper_x)]
        local_hand_with_work = img_hand_with_work[int(lower_y):int(upper_y),int(lower_x):int(upper_x)]
        local_hand = img_hand[int(lower_y):int(upper_y),int(lower_x):int(upper_x)]
        #250×250→125×125に圧縮
        size = (OUTPUT_HEIGHT , OUTPUT_WIDTH)#chainerへの入力となるサイズ
        compressed_depth = cv2.resize(local_depth,size)
        cv2.imwrite('./Method{}/EXPERIMENT/{}/QUARTER_DEPTH/img{}.png'.format(METHOD,tstr,top_count),compressed_depth)
        #
        compressed_hand_with_work = cv2.resize(local_hand_with_work,size)
        cv2.imwrite('./Method{}/EXPERIMENT/{}/QUARTER_HAND_WITH_WORK/img{}.png'.format(METHOD,tstr,top_count),compressed_hand_with_work)
        #
        compressed_hand = cv2.resize(local_hand,size)
        cv2.imwrite('./Method{}/EXPERIMENT/{}/QUARTER_HAND/img{}.png'.format(METHOD,tstr,top_count),compressed_hand)

 #Graspabilityの高い把持位置"TOP_NUMBER"箇所を１枚に描画
#１位.赤色、２位以降.黄色
def draw_best_hand_with_work(best_pass,B,G,R):
    img = cv2.imread('{}'.format(best_pass),1)  #※グレーで読み込み（0）すると，円や直線も白か黒しか描けなくなるので注意
    #
    for top_count in range(TOP_NUMBER):
        ImgX = Top_ImgX[top_count]
        ImgY = Top_ImgY[top_count]
        Angle = Top_Angle[top_count]
        #finger_topは画像上で上側の指で，finger_bottomは画像上で下側の指
        finger_top_x = int( ImgX + ( ( HAND_WIDTH / 2 ) * BEFORE_TO_AFTER * math.cos(Angle) ) )
        finger_top_y = int( ImgY - ( ( HAND_WIDTH / 2 ) * BEFORE_TO_AFTER * math.sin(Angle) ) )
        finger_bottom_x = int( ImgX - ( ( HAND_WIDTH / 2 ) * BEFORE_TO_AFTER * math.cos(Angle) ) )
        finger_bottom_y = int( ImgY + ( ( HAND_WIDTH / 2 ) * BEFORE_TO_AFTER * math.sin(Angle) ) )
        #ハンドをカラーで描画
        if top_count ==0:
            cv2.circle(img, (finger_top_x, finger_top_y), 10, (0, 0, 255), -1)
            cv2.circle(img, (finger_bottom_x, finger_bottom_y), 10, (0, 0, 255), -1)
            cv2.line(img, (finger_top_x, finger_top_y), (finger_bottom_x, finger_bottom_y), (0, 0, 255), 5)
        else :
            cv2.circle(img, (finger_top_x, finger_top_y), 10, (int(B*(TOP_NUMBER-top_count)/(TOP_NUMBER-1)), int(G*(TOP_NUMBER-top_count)/(TOP_NUMBER-1)), int(R*(TOP_NUMBER-top_count)/(TOP_NUMBER-1))), -1)
            cv2.circle(img, (finger_bottom_x, finger_bottom_y), 10, (int(B*(TOP_NUMBER-top_count)/(TOP_NUMBER-1)), int(G*(TOP_NUMBER-top_count)/(TOP_NUMBER-1)), int(R*(TOP_NUMBER-top_count)/(TOP_NUMBER-1))), -1)
            cv2.line(img, (finger_top_x, finger_top_y), (finger_bottom_x, finger_bottom_y), (int(B*(TOP_NUMBER-top_count)/(TOP_NUMBER-1)), int(G*(TOP_NUMBER-top_count)/(TOP_NUMBER-1)), int(R*(TOP_NUMBER-top_count)/(TOP_NUMBER-1))), 5)
    #保存
    cv2.imwrite('{}'.format(best_pass),img)

def SuccessFailureCheck(METHOD,tstr,img_pass,final_center_x,final_center_y,final_center_z,final_angle,black_pass):
    '''
    # ハンド画像を作成 ==※これだと、ハンドがはみ出てしまう場合でも中央に描画してしまうのでNG（2018.9.16）
    img_hand = cv2.imread('./black250.png',0)  #グレーで読み込み
    finger_top_x = int( ( 250 / 2 ) + ( ( HAND_WIDTH / 2 ) * BEFORE_TO_AFTER * math.cos(final_angle) ) )
    finger_top_y = int( ( 250 / 2 ) - ( ( HAND_WIDTH / 2 ) * BEFORE_TO_AFTER * math.sin(final_angle) ) )
    finger_bottom_x = int( ( 250 / 2 ) - ( ( HAND_WIDTH / 2 ) * BEFORE_TO_AFTER * math.cos(final_angle) ) )
    finger_bottom_y = int( ( 250 / 2 ) + ( ( HAND_WIDTH / 2 ) * BEFORE_TO_AFTER * math.sin(final_angle) ) )
    cv2.circle(img_hand, (finger_top_x, finger_top_y), 10, (255, 255, 255), -1)
    cv2.circle(img_hand, (finger_bottom_x, finger_bottom_y), 10, (255, 255, 255), -1)
    cv2.line(img_hand, (finger_top_x, finger_top_y), (finger_bottom_x, finger_bottom_y), (255, 255, 255), 5)
    size = (OUTPUT_HEIGHT , OUTPUT_WIDTH)   #chainerへの入力となるサイズ
    compressed_img_hand = cv2.resize(img_hand,size)
    cv2.imwrite('./Method{}/EXPERIMENT/{}/M1tmp/quarter_hand.png'.format(METHOD,tstr),compressed_img_hand)
    '''
    # ハンド画像を作成 ==
    img_hand = cv2.imread('{}'.format(black_pass),0)  #グレーで読み込み
    finger_top_x = int( ( final_center_x ) + ( ( HAND_WIDTH / 2 ) * BEFORE_TO_AFTER * math.cos(final_angle) ) )
    finger_top_y = int( ( final_center_y ) - ( ( HAND_WIDTH / 2 ) * BEFORE_TO_AFTER * math.sin(final_angle) ) )
    finger_bottom_x = int( ( final_center_x ) - ( ( HAND_WIDTH / 2 ) * BEFORE_TO_AFTER * math.cos(final_angle) ) )
    finger_bottom_y = int( ( final_center_y ) + ( ( HAND_WIDTH / 2 ) * BEFORE_TO_AFTER * math.sin(final_angle) ) )
    # ハンドを白色描画
    cv2.circle(img_hand, (finger_top_x, finger_top_y), 10, (255, 255, 255), -1)
    cv2.circle(img_hand, (finger_bottom_x, finger_bottom_y), 10, (255, 255, 255), -1)
    cv2.line(img_hand, (finger_top_x, finger_top_y), (finger_bottom_x, finger_bottom_y), (255, 255, 255), 5)
    cv2.imwrite('./Method{}/EXPERIMENT/{}/M1tmp/hand.png'.format(METHOD,tstr),img_hand)
    #x座標のはみ出し判定
    if final_center_x - (WINDOW_WIDTH // 2) <= 0:
        lower_x = 0
        upper_x = WINDOW_WIDTH
    elif TRIMMED_WIDTH <= final_center_x + (WINDOW_WIDTH // 2) :
        lower_x = TRIMMED_WIDTH - WINDOW_WIDTH
        upper_x = TRIMMED_WIDTH
    else:
        lower_x = final_center_x - (WINDOW_WIDTH // 2)
        upper_x = final_center_x + (WINDOW_WIDTH // 2)
    #y座標のはみ出し判定
    if final_center_y - (WINDOW_HEIGHT // 2) <= 0:
        lower_y = 0
        upper_y = WINDOW_WIDTH
    elif TRIMMED_HEIGHT <= final_center_y + (WINDOW_HEIGHT // 2) :
        lower_y = TRIMMED_HEIGHT - WINDOW_HEIGHT
        upper_y = TRIMMED_HEIGHT
    else:
        lower_y = final_center_y - (WINDOW_HEIGHT // 2)
        upper_y = final_center_y + (WINDOW_HEIGHT // 2)

    #500×500→250×250をそのまま切り取った後、圧縮
    local_hand = img_hand[int(lower_y):int(upper_y),int(lower_x):int(upper_x)]
    size = (OUTPUT_HEIGHT , OUTPUT_WIDTH)#chainerへの入力となるサイズ
    compressed_hand = cv2.resize(local_hand,size)
    cv2.imwrite('./Method{}/EXPERIMENT/{}/M1tmp/quarter_hand.png'.format(METHOD,tstr),compressed_hand)


    # hand_with_work画像を用作成 ==
    img_hand_with_work = cv2.imread('{}'.format(img_pass),0)  #グレーで読み込み
    finger_top_x = int( ( final_center_x ) + ( ( HAND_WIDTH / 2 ) * BEFORE_TO_AFTER * math.cos(final_angle) ) )
    finger_top_y = int( ( final_center_y ) - ( ( HAND_WIDTH / 2 ) * BEFORE_TO_AFTER * math.sin(final_angle) ) )
    finger_bottom_x = int( ( final_center_x ) - ( ( HAND_WIDTH / 2 ) * BEFORE_TO_AFTER * math.cos(final_angle) ) )
    finger_bottom_y = int( ( final_center_y ) + ( ( HAND_WIDTH / 2 ) * BEFORE_TO_AFTER * math.sin(final_angle) ) )
    # ハンドを白色描画
    cv2.circle(img_hand_with_work, (finger_top_x, finger_top_y), 10, (255, 255, 255), -1)
    cv2.circle(img_hand_with_work, (finger_bottom_x, finger_bottom_y), 10, (255, 255, 255), -1)
    cv2.line(img_hand_with_work, (finger_top_x, finger_top_y), (finger_bottom_x, finger_bottom_y), (255, 255, 255), 5)
    cv2.imwrite('./Method{}/EXPERIMENT/{}/M1tmp/hand_with_work.png'.format(METHOD,tstr),img_hand_with_work)
    #x座標のはみ出し判定
    if final_center_x - (WINDOW_WIDTH // 2) <= 0:
        lower_x = 0
        upper_x = WINDOW_WIDTH
    elif TRIMMED_WIDTH <= final_center_x + (WINDOW_WIDTH // 2) :
        lower_x = TRIMMED_WIDTH - WINDOW_WIDTH
        upper_x = TRIMMED_WIDTH
    else:
        lower_x = final_center_x - (WINDOW_WIDTH // 2)
        upper_x = final_center_x + (WINDOW_WIDTH // 2)
    #y座標のはみ出し判定
    if final_center_y - (WINDOW_HEIGHT // 2) <= 0:
        lower_y = 0
        upper_y = WINDOW_WIDTH
    elif TRIMMED_HEIGHT <= final_center_y + (WINDOW_HEIGHT // 2) :
        lower_y = TRIMMED_HEIGHT - WINDOW_HEIGHT
        upper_y = TRIMMED_HEIGHT
    else:
        lower_y = final_center_y - (WINDOW_HEIGHT // 2)
        upper_y = final_center_y + (WINDOW_HEIGHT // 2)
    #500×500→250×250をそのまま切り取った後、圧縮
    local_hand_with_work = img_hand_with_work[int(lower_y):int(upper_y),int(lower_x):int(upper_x)]
    size = (OUTPUT_HEIGHT , OUTPUT_WIDTH)#chainerへの入力となるサイズ
    compressed_hand_with_work = cv2.resize(local_hand_with_work,size)
    cv2.imwrite('./Method{}/EXPERIMENT/{}/M1tmp/quarter_hand_with_work.png'.format(METHOD,tstr),compressed_hand_with_work)

    # デプス画像を用作成 ==
    img_depth = cv2.imread('{}'.format(img_pass),0)  #グレーで読み込み
    #x座標のはみ出し判定
    if final_center_x - (WINDOW_WIDTH // 2) <= 0:
        lower_x = 0
        upper_x = WINDOW_WIDTH
    elif TRIMMED_WIDTH <= final_center_x + (WINDOW_WIDTH // 2) :
        lower_x = TRIMMED_WIDTH - WINDOW_WIDTH
        upper_x = TRIMMED_WIDTH
    else:
        lower_x = final_center_x - (WINDOW_WIDTH // 2)
        upper_x = final_center_x + (WINDOW_WIDTH // 2)
    #y座標のはみ出し判定
    if final_center_y - (WINDOW_HEIGHT // 2) <= 0:
        lower_y = 0
        upper_y = WINDOW_WIDTH
    elif TRIMMED_HEIGHT <= final_center_y + (WINDOW_HEIGHT // 2) :
        lower_y = TRIMMED_HEIGHT - WINDOW_HEIGHT
        upper_y = TRIMMED_HEIGHT
    else:
        lower_y = final_center_y - (WINDOW_HEIGHT // 2)
        upper_y = final_center_y + (WINDOW_HEIGHT // 2)
    #500×500→250×250をそのまま切り取った後、圧縮
    local_depth = img_depth[int(lower_y):int(upper_y),int(lower_x):int(upper_x)]
    size = (OUTPUT_HEIGHT , OUTPUT_WIDTH)#chainerへの入力となるサイズ
    compressed_depth = cv2.resize(local_depth,size)
    cv2.imwrite('./Method{}/EXPERIMENT/{}/M1tmp/quarter_depth.png'.format(METHOD,tstr),compressed_depth)

#各種設定=============================================
#INPUT_HEIGHT = 638  #入力深度画像の縦
#INPUT_WIDTH = 588   #入力深度画像の横
OUTPUT_HEIGHT = 125  #出力深度画像（CNNへの入力）の縦
OUTPUT_WIDTH = 125   #出力深度画像（CNNへの入力）の横
#2Dキャリブレーションなら内側底面を切り取るようにマージンを設定
#3Dキャリブレーションなら外側上面を切り取るようにマージンを設定
LEFT_MARGIN = 321    #箱の縁などが入力画像に写ってしまう場合，左をどれだけ無視するか
RIGHT_MARGIN = 1280-910  #箱の縁などが入力画像に写ってしまう場合，右をどれだけ無視するか
TOP_MARGIN = 263     #箱の縁などが入力画像に写ってしまう場合，上をどれだけ無視するか
BOTTOM_MARGIN = 1024-886  #箱の縁などが入力画像に写ってしまう場合，下をどれだけ無視するか
BETANURI_BLACK = 15 #トリミング後の画像（箱外枠含む）に、BETANURI_BLACKピクセル幅の黒枠を描画することで枠を消す
# Method0の設定=
DismissAreaWidth_Graspability = 50 # Graspabilityの探索から除外する（上下左右からの）エリア幅
# Method1の設定
WINDOW_HEIGHT = 250 #ラスタ走査するウィンドウの縦
WINDOW_WIDTH = 250  #ラスタ走査するウィンドウの横
WINDOW_STRIDE = 50  #ラスタ走査するウィンドウのずらし幅
DissmissPixelValue = 100 #画像の中央がこれ以下の輝度値ならスキップ
#全体に関わる設定
BEFORE_TO_AFTER = 500 / 225  #( シミュレーションでのdepth画像のサイズ ) / (　シミュレーション内での箱のサイズ　)
HAND_WIDTH=40 #ハンドの開き幅 //シミュレーションと同様に40mm　ただし、Graspability計算時のハンドの形状(HAND_THICKNESS)は要調整
TOP_NUMBER = 5
CATCH_DEPTH_WINDOW = 20 #実際の距離値の取得は把持中心位置近傍の深度画像値の最大値から算出する
#カメラ関連===========================================================
#次の二行は　OniSampleUtilities.hの定義ではmm こっちではm で記述することに注意！！
MAX_DISTANCE = 960*0.001 #WindowsPCのmain.cpp（PLY→デプス画像）の定義と同じになっているか要確認 ※ただし単位をmm→Mに変換
MIN_DISTANCE = 900*0.001 #WindowsPCのmain.cpp（PLY→デプス画像）の定義と同じになっているか要確認 ※ただし単位をmm→Mに変換
StandardPointX = 0.63 #キャリブレーション時の最終コーナー（南東）のxy座標
StandardPointY = 0.10 #キャリブレーション時の最終コーナー（南東）のxy座標
BoxX = 0.23 #使用する箱の長さ（世界座標X方向）
BoxY = 0.22 #使用する箱の長さ（世界座標Y方向）
#CNNのモデルを設定===================================================================
M1_model = CNN1()
Path = "./Method1/SuccessFailure_Work4.pth"
param = torch.load(Path)
M1_model.load_state_dict(param) 
M1_model.eval()
# serializers.load_npz("./Method1/SuccessFailure_Work1.model10",M1_model)

# main
if __name__ == "__main__":
    #parser = argparse.ArgumentParser(description='Learning Grasp_position and Graspability from depth image')
    #parser.add_argument('--gpu', '-gpu', default=-1, type=int,help='GPU ID (negative value indicates CPU)')  #GPU:0 CPU:-1
    #args = parser.parse_args()
    #if args.gpu >= 0:
        #cuda.check_cuda_available()
    #xp = cuda.cupy if args.gpu >= 0 else np
    xp = np

    #端末からの引数を取得
    argv = sys.argv
    argc = len(argv)
    if(argc != 2):
        print("Error : there are many argument.")
    #１つ目の引数（arg[0])はプログラム名なので無視
    #２つ目の引数はMethod番号（0→M0, 1→M1, 2→M3 を実行）
    METHOD = int(argv[1])

    tdatetime = dt.now()
    tstr = tdatetime.strftime('RESULT%Y-%m-%d_%H_%M_%S')
    # im = cv2.imread('./DEPTH_TMP/MedianFilterDepth_Rotated.png',0)
    im = cv2.imread('./DEPTH_TMP/reshape.png',0)
    #im = cv2.imread('./DEPTH_TMP/MedianFilterDepth_Rotated.png',0)
    #画像サイズを取得し、設定したマージンから、トリミング後のサイズを計算
    INPUT_HEIGHT, INPUT_WIDTH = im.shape
    TRIMMED_HEIGHT= 623
    TRIMMED_WIDTH= 659
    im_cut = np.zeros((TRIMMED_HEIGHT, TRIMMED_WIDTH))
    height_mergin = (TRIMMED_HEIGHT - INPUT_HEIGHT)//2
    width_mergin = (TRIMMED_WIDTH - INPUT_WIDTH)//2
    im_cut[height_mergin:height_mergin+INPUT_HEIGHT, width_mergin:width_mergin+INPUT_WIDTH] = im

    #さらに画像サイズはそのままで、輪郭を黒塗りにする（上下左右 数ピクセルを黒色化※一番最後の数字が黒色にしたいピクセル幅)
    cv2.rectangle(im_cut,(0,0),(TRIMMED_WIDTH,TRIMMED_HEIGHT),(0,0,0),BETANURI_BLACK)

    #トリミング後と同じサイズの黒一色画像を用意(Method0,2実行時の最終的な把持位置を成否CNNで予測する際に使用)
    black = cv2.imread('./black1280_1024.png',0)
    black_cut = black[0:TRIMMED_HEIGHT, 0:TRIMMED_WIDTH]

    if METHOD == 0:
        # 名前が『年月日時分秒』 のフォルダが作成されます
        os.makedirs("./Method{}/EXPERIMENT/{}".format(METHOD,tstr))
        os.makedirs("./Method{}/EXPERIMENT/{}/QUARTER_DEPTH".format(METHOD,tstr))
        os.makedirs("./Method{}/EXPERIMENT/{}/QUARTER_HAND".format(METHOD,tstr))
        os.makedirs("./Method{}/EXPERIMENT/{}/QUARTER_HAND_WITH_WORK".format(METHOD,tstr))
        os.makedirs("./Method{}/EXPERIMENT/{}/HAND_WITH_WORK".format(METHOD,tstr))
        os.makedirs("./Method{}/EXPERIMENT/{}/tmp".format(METHOD,tstr))
        #成功予測された把持位置を記録するテキスト
        fp = open('./Method{}/EXPERIMENT/{}/best{}.txt'.format(METHOD,tstr,TOP_NUMBER),'wt')
        fp.close()
        cv2.imwrite('./Method{}/EXPERIMENT/{}/depth.png'.format(METHOD,tstr),im_cut)
        img_pass = './Method{}/EXPERIMENT/{}/depth.png'.format(METHOD,tstr)
        cv2.imwrite('./Method{}/EXPERIMENT/{}/black.png'.format(METHOD,tstr),black_cut)
        black_pass = './Method{}/EXPERIMENT/{}/black.png'.format(METHOD,tstr)
        cv2.imwrite('./Method{}/EXPERIMENT/{}/best{}.png'.format(METHOD,tstr,TOP_NUMBER),im_cut)
        best_pass = './Method{}/EXPERIMENT/{}/best{}.png'.format(METHOD,tstr,TOP_NUMBER)

        #探索時間の計測開始1
        StartTime=time.time()

        #Graspabilityを計算
        Graspability(METHOD,tstr,img_pass)

        #探索時間の計測終了1
        DurationTime = time.time() - StartTime
        print("Graspabilityの探索時間は{}secでした。".format(DurationTime))

        #CNN用の準備（引数となる画像から局所領域を切り取り保存×TOP_NUMBER回）
        M0_Draw_and_LocalCut_and_Compress(METHOD,tstr,img_pass)
        #Graspabilityの高い把持位置をカラーで描画
        draw_best_hand_with_work(best_pass,0,255,255)

        for top_count in range(TOP_NUMBER):
            #とりあえずログを記録
            fp = open('./Method{}/EXPERIMENT/{}/best{}.txt'.format(METHOD,tstr,TOP_NUMBER),'a')
            print("G{}位".format(top_count+1),file=fp)
            print("Graspability={:.3f} / Gマップ=G{:.0f}_{:.0f} / (x,y)=({:.0f},{:.0f}) / 回転角={:.2f}rad / 深度(0-255)={:.0f}".format(Top_Graspability[top_count]/255,Top_CountA[top_count],Top_CountB[top_count],Top_ImgX[top_count],Top_ImgY[top_count],Top_Angle[top_count],Top_ImgZ[top_count]),file=fp)
            #print("Gマップ=G{}_{} / (x,y)=({}, {}) / 回転角={}rad / 深度（0〜255）={}".format(Top_CountA[top_count],Top_CountB[top_count],Top_ImgX[top_count],Top_ImgY[top_count],Top_Angle[top_count],Top_ImgZ[top_count]),file=fp)
            print("====================================================================================================================================",file=fp)
            fp.close()

        #最終的に確定した把持位置の「QUARTER_DEPTH画像」「QUARTER_HAND画像」「QUARTER_HAND_WITH_WORK画像」をコピー
        final_quarter_depth = cv2.imread("./Method{}/EXPERIMENT/{}/QUARTER_DEPTH/img0.png".format(METHOD,tstr),0)
        cv2.imwrite('./Method{}/EXPERIMENT/{}/final_quarter_depth.png'.format(METHOD,tstr),final_quarter_depth)
        final_quarter_hand = cv2.imread("./Method{}/EXPERIMENT/{}/QUARTER_HAND/img0.png".format(METHOD,tstr),0)
        cv2.imwrite('./Method{}/EXPERIMENT/{}/final_quarter_hand.png'.format(METHOD,tstr),final_quarter_hand)
        final_quarter_hand_with_work = cv2.imread("./Method{}/EXPERIMENT/{}/QUARTER_HAND_WITH_WORK/img0.png".format(METHOD,tstr),0)
        cv2.imwrite('./Method{}/EXPERIMENT/{}/final_quarter_hand_with_work.png'.format(METHOD,tstr),final_quarter_hand_with_work)

        #最適把持位置を確定
        final_top_count = 0
        final_center_x = Top_ImgX[final_top_count]
        final_center_y = Top_ImgY[final_top_count]
        final_center_z = Top_ImgZ[final_top_count]
        final_angle = Top_Angle[final_top_count]

        # #CNN用の準備（引数となる画像から局所領域を切り取り保存×TOP_NUMBER回）
        # draw_and_local_cut_and_compress(METHOD,tstr,img_pass)

        if LOG_MODE == 1:
            print("log作成中..")
            #M1でも推定
            #M1====================================================================================
            os.makedirs("./Method{}/EXPERIMENT/{}/M1tmp".format(METHOD,tstr))
            SuccessFailureCheck(METHOD,tstr,img_pass,final_center_x,final_center_y,final_center_z,final_angle,black_pass)
            #深度画像読み込み
            img_a=np.array(Image.open('./Method{}/EXPERIMENT/{}/M1tmp/quarter_depth.png'.format(METHOD,tstr)).convert('L'),'f')
            img_a /= 255
            img_a=img_a.reshape(1,1,OUTPUT_HEIGHT,OUTPUT_WIDTH)
            img_a=img_a.astype(np.float32)
            #ハンド画像読み込み
            img_b=np.array(Image.open('./Method{}/EXPERIMENT/{}/M1tmp/quarter_hand.png'.format(METHOD,tstr)).convert('L'),'f')
            img_b/=255
            img_b=img_b.reshape(1,1,OUTPUT_HEIGHT,OUTPUT_WIDTH)
            img_b=img_b.astype(np.float32)
            SuccessProbability = M1_model.output(img_a,img_b).reshape(2,1)[1]
            SuccessProbability.astype(np.float32)
            #======================================================================================
            fp = open('./Method{}/EXPERIMENT/{}/log.txt'.format(METHOD,tstr),'wt')
            print("Graspability={:.3f} / 成否予測CNN確率={:.3f}".format(Top_Graspability[0]/255,SuccessProbability[0]),file=fp)
            fp.close()

    if METHOD == 1:
        # 名前が『年月日時分秒』 のフォルダが作成されます
        os.makedirs("./Method{}/EXPERIMENT/{}".format(METHOD,tstr))
        os.makedirs("./Method{}/EXPERIMENT/{}/QUARTER_DEPTH".format(METHOD,tstr))
        os.makedirs("./Method{}/EXPERIMENT/{}/QUARTER_HAND".format(METHOD,tstr))
        os.makedirs("./Method{}/EXPERIMENT/{}/QUARTER_HAND_WITH_WORK".format(METHOD,tstr))
        os.makedirs("./Method{}/EXPERIMENT/{}/HAND_WITH_WORK".format(METHOD,tstr))
        #成功予測された把持位置を記録するテキスト
        fp = open('./Method{}/EXPERIMENT/{}/best{}.txt'.format(METHOD,tstr,TOP_NUMBER),'wt')
        fp.close()
        cv2.imwrite('./Method{}/EXPERIMENT/{}/depth.png'.format(METHOD,tstr),im_cut)
        img_pass = './Method{}/EXPERIMENT/{}/depth.png'.format(METHOD,tstr)
        cv2.imwrite('./Method{}/EXPERIMENT/{}/black.png'.format(METHOD,tstr),black_cut)
        black_pass = './Method{}/EXPERIMENT/{}/black.png'.format(METHOD,tstr)
        cv2.imwrite('./Method{}/EXPERIMENT/{}/best{}.png'.format(METHOD,tstr,TOP_NUMBER),im_cut)
        best_pass = './Method{}/EXPERIMENT/{}/best{}.png'.format(METHOD,tstr,TOP_NUMBER)
        #os.makedirs("./Method{}/EXPERIMENT/{}/HAND".format(tstr))#ハンドのテンプレート（最初に作成して再利用することで無駄な処理を無くす）
        #ハンドのテンプレートを回転角の数だけ作成
        # for k in range(0,ANGLE_NUMBER):
        #     angle=(math.pi/ANGLE_NUMBER)*k
        #     draw_hand(METHOD,tstr,angle,k)
        # for x in range(int( center_x-(CATCH_DEPTH_WINDOW/2) ), int( center_x+(CATCH_DEPTH_WINDOW/2) )):
        #     for y in range(int( center_y-(CATCH_DEPTH_WINDOW/2) ),int( center_y+(CATCH_DEPTH_WINDOW/2) )):
        #         #print('各輝度値：{}'.format(pixels[x,y][0]))
        #         if(pixels[x,y][0] > MaxDepthPixelValue):
        #             MaxDepthPixelValue = pixels[x,y][0]

        #全てのログを記録するテキスト
        fp = open('./Method{}/EXPERIMENT/{}/HAND_WITH_WORK/estimation.txt'.format(METHOD,tstr),'wt')
        fp.close()
        count = 0
        #local_depth_count = 0   #LOCALのDEPTH画像はウィンドウごと（つまり，回転角が変わるごとに作成する必要はない）
        #全候補の値を格納しておく（初期化）
        all_center_x = np.array([])
        all_center_y = np.array([])
        all_center_z = np.array([])#輝度値
        all_angle = np.array([])
        all_SuccessProbability = np.array([])
        #もし全ての候補が無視された場合エラーとなってしまうので，その際は"No Graspable Position"と表示
        #そのためのフラグ
        NothingFlag = 0
        #探索時間を計測
        StartTime=time.time()
        ANGLE_NUMBER = 4
        for y in range(0,TRIMMED_HEIGHT - WINDOW_HEIGHT + 1, WINDOW_STRIDE): #ラスタ走査する各ウィンドウ左上座標（縦）
            for x in range(0,TRIMMED_WIDTH - WINDOW_WIDTH + 1, WINDOW_STRIDE): #ラスタ走査する各ウィンドウ左上座標（横）
                #把持位置中心座標の輝度値が閾値以下（黒色）ならばCNNに通さない
                center_y = int(y + ( WINDOW_HEIGHT / 2 ))    #把持位置中心座標
                center_x = int(x + ( WINDOW_WIDTH / 2 ))     #把持位置中心座標
                GrobalDepth=Image.open('{}'.format(img_pass))  #※グレーで読み込み
                pixels = GrobalDepth.load()
                if(pixels[center_x,center_y]>=DissmissPixelValue):
                    NothingFlag = 1
                    #近傍のみ切り抜いたdepth画像は角度の変わる毎ではなく，ラスタウィンドウが変わるごとで良い
                    M1_LocalCut_and_Compress(METHOD,tstr,img_pass,x,y,count)  #入力画像からラスタウィンドウを切り取り，chainerへの入力サイズに圧縮
                    #img_a=np.array(Image.open('./Method{}/EXPERIMENT/{}/QUARTER_DEPTH/img{}_{}.bmp'.format(METHOD,tstr,count,count+ANGLE_NUMBER-1)).convert('L'),'f')
                    img_a=np.array(Image.open('./Method{}/EXPERIMENT/{}/QUARTER_DEPTH/img{}_{}.png'.format(METHOD,tstr,count,count+ANGLE_NUMBER-1)).convert('L'),'f')
                    img_a /= 255
                    img_a=img_a.reshape(1,1,OUTPUT_HEIGHT,OUTPUT_WIDTH)
                    img_a=img_a.astype(np.float32)
                    for k in range(0,ANGLE_NUMBER):
                        angle=(math.pi/ANGLE_NUMBER)*k
                        #center_y = y + ( WINDOW_HEIGHT / 2 )    #把持位置中心座標
                        #center_x = x + ( WINDOW_WIDTH / 2 )     #把持位置中心座標
                        #深度画像中央と同じ輝度値でハンドを描画し保存
                        GripperD = 50
                        #center_z = GripperD/2
                        if pixels[center_x,center_y] <= GripperD*1.5:
                            center_z = GripperD/2
                        else:
                            center_z = pixels[center_x,center_y] - GripperD
                        all_center_z = np.append(all_center_z,center_z)
                        M1_DrawHand_and_Compress(METHOD,tstr,angle,count)
                        img_b=np.array(Image.open('./Method{}/EXPERIMENT/{}/QUARTER_HAND/img{}.png'.format(METHOD,tstr,count)).convert('L'),'f')
                        #img_b=np.array(Image.open('./Method{}/EXPERIMENT/{}/QUARTER_HAND/img{}.png'.format(METHOD,tstr,k)).convert('L'),'f')
                        #ハンド画像を濃淡で描画
                        #img_b = (img_b/255)*pixels[center_x,center_y][0]
                        img_b/=255
                        img_b=img_b.reshape(1,1,OUTPUT_HEIGHT,OUTPUT_WIDTH)
                        img_b=img_b.astype(np.float32)
                        SuccessProbability = M1_model.output(img_a,img_b).reshape(2,1)[1]
                        SuccessProbability.float()
                        #全候補の値を格納しておく
                        all_center_x=np.append(all_center_x,center_x)
                        all_center_y=np.append(all_center_y,center_y)
                        all_angle=np.append(all_angle,angle)
                        all_SuccessProbability=np.append(all_SuccessProbability,SuccessProbability)
                        #とりあえずログを記録
                        fp = open('./Method{}/EXPERIMENT/{}/HAND_WITH_WORK/estimation.txt'.format(METHOD,tstr),'a')
                        print("hand{}   estimated SuccessProbability = {:.3f}".format(count,SuccessProbability[0]),file=fp)
                        fp.close()
                        M1_DrawHandWithWork_and_Compress(METHOD,tstr,img_pass,center_x,center_y,angle,count) #入力画像に把持位置を描画
                        count +=1

        #all_graspability=「（インデックス0のGraspability），（インデックス1のGraspability）,..」
        #sorted_index_of_graspability=「（Graspability最大のインデックス），（Graspability2番目のインデックス）,...」
        #sorted_index_of_graspability = np.array(all_graspability)
        #print(sorted_index_of_graspability)
        #sorted_index_of_graspability.argsort()[::-1]
        #print(sorted_index_of_graspability)

        #探索時間を表示
        DurationTime = time.time() - StartTime
        print("探索時間は{}secでした。".format(DurationTime))
        #2017.10.25 修正箇所 3 of 3################################################################################################################################################################
        if (NothingFlag==0):
            print("Failure (No Graspable Position)")
        elif(NothingFlag==1):
            print("Success (Found Graspable Positon)")
            sorted_index_of_SuccessProbability = np.argsort(all_SuccessProbability)[::-1]
        ###########################################################################################################################################################################################
            #上位TOP_NUMBER個を描画
            for top_count in range(TOP_NUMBER):

                index = int( sorted_index_of_SuccessProbability[top_count] )
                SuccessProbability = all_SuccessProbability[index]
                center_x = all_center_x[index]
                center_y = all_center_y[index]
                center_z = all_center_z[index]
                angle = all_angle[index]

                #最適位置は赤色で描画！！
                if top_count == 0 :
                    f_best = open('./Method{}/EXPERIMENT/{}/best{}.txt'.format(METHOD,tstr,TOP_NUMBER),'a')
                    #print("最適把持位置\n",file=f_best)
                    print("index={} center_x={} center_y={} angle={} SuccessProbability={}".format(index,center_x,center_y,angle,SuccessProbability),file=f_best)
                    f_best.close()
                    M1_DrawBestHandWithWork(best_pass,center_x,center_y,angle,0,0,255)   #最後の3引数は描画する"BGR"の値
                    final_top_count = index
                    #最適なQUARTER_EPTH画像をコピー
                    #(インデックスをANGLE_NUMBERで割った商　×　ANGLE_NUMBER )から(それ+ANGLE_NUMBER-1まで)
                    #shutil.copyfile('./OTHERS/{}/ESTIMATION/QUARTER_DEPTH/img{}_{}.bmp'.format(tstr,int((index//ANGLE_NUMBER)*ANGLE_NUMBER),int((index//ANGLE_NUMBER)*ANGLE_NUMBER+ANGLE_NUMBER-1)),'./QUARTER_DEPTH/img{}.bmp'.format(tstr))
                    #最適なQUARTER_HAND画像をコピー
                    #インデックスをANGLE_NUMBERで割った余りがハンドのテンプレのインデックスと一致
                    #shutil.copyfile('./OTHERS/{}/HAND/img{}.png'.format(tstr,index%ANGLE_NUMBER),'./QUARTER_HAND/img{}.png'.format(tstr))

                #2番目以降は徐々に薄くなる黄色で描画！！
                else :
                    f_best = open('./Method{}/EXPERIMENT/{}/best{}.txt'.format(METHOD,tstr,TOP_NUMBER),'a')
                    print("index={} center_x={} center_y={} angle={} SuccessProbability={}".format(index,center_x,center_y,angle,SuccessProbability),file=f_best)
                    f_best.close()
                    RED = int( 255 - ( ( 255 - 50 ) / ( TOP_NUMBER - 1) ) * ( top_count - 1 ) ) #輝度値255から50の間で徐々に薄くなるように
                    GREEN = int( 255 - ( ( 255 - 50 ) / ( TOP_NUMBER - 1) ) * ( top_count - 1 ) ) #輝度値255から50の間で徐々に薄くなるように
                    BLUE = 0
                    M1_DrawBestHandWithWork(best_pass,center_x,center_y,angle,BLUE,GREEN,RED)#把持可能群は白で描画

            #最終的に確定した把持位置の「QUARTER_DEPTH画像」「QUARTER_HAND画像」「QUARTER_HAND_WITH_WORK画像」をコピー
            print(int((final_top_count//ANGLE_NUMBER)*ANGLE_NUMBER))
            final_quarter_depth = cv2.imread("./Method{}/EXPERIMENT/{}/QUARTER_DEPTH/img{}_{}.png".format(METHOD,tstr,int((final_top_count//ANGLE_NUMBER)*ANGLE_NUMBER),int((final_top_count//ANGLE_NUMBER)*ANGLE_NUMBER+ANGLE_NUMBER-1)),0)
            cv2.imwrite('./Method{}/EXPERIMENT/{}/final_quarter_depth.png'.format(METHOD,tstr),final_quarter_depth)
            final_quarter_hand = cv2.imread("./Method{}/EXPERIMENT/{}/QUARTER_HAND/img{}.png".format(METHOD,tstr,final_top_count),0)
            cv2.imwrite('./Method{}/EXPERIMENT/{}/final_quarter_hand.png'.format(METHOD,tstr),final_quarter_hand)
            final_quarter_hand_with_work = cv2.imread("./Method{}/EXPERIMENT/{}/QUARTER_HAND_WITH_WORK/img{}.png".format(METHOD,tstr,final_top_count),0)
            cv2.imwrite('./Method{}/EXPERIMENT/{}/final_quarter_hand_with_work.png'.format(METHOD,tstr),final_quarter_hand_with_work)

            #最適把持位置
            final_center_x = all_center_x[final_top_count]
            final_center_y = all_center_y[final_top_count]
            final_center_z = all_center_z[final_top_count]
            final_angle = all_angle[final_top_count]

            if LOG_MODE == 1:
                print("log作成中..")
                #M0でも推定
                #M0====================================================================================
                os.makedirs("./Method{}/EXPERIMENT/{}/M0tmp".format(METHOD,tstr))
                M1_Graspability(METHOD,tstr,img_pass,final_center_x,final_center_y,final_center_z,final_angle)
                
                #======================================================================================
                fp = open('./Method{}/EXPERIMENT/{}/log.txt'.format(METHOD,tstr),'wt')
                print("Graspability={:.3f} / 成否予測CNN確率={:.3f}".format(_Graspability_M1/255,all_SuccessProbability[final_top_count]),file=fp)
                fp.close()
