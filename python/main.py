import random
import cv2
import numpy as np
import time

def getCnt(img):
    #HSV変換
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    mask = np.zeros(h.shape, dtype=np.uint8)
    #教科書の領域を白、背景を黒で二値化
    mask[(v > 68)] = 255
    #ノイズの除去
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask,kernel)
    mask = cv2.dilate(mask,kernel)
    mask = cv2.dilate(mask,kernel)
    mask = cv2.erode(mask,kernel)
    #領域抽出
    contours, hierarchy= cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = list(filter(lambda cnt: 1 < cv2.contourArea(cnt), contours))
    MaxArea = 0
    MaxCnt = 0
    cv2.imwrite("../photo/other/bin_getRect.jpg",mask)
    return contours

def getVertex(cnt,img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    Amask = np.zeros(gray.shape, dtype=np.uint8)
    Cmask = np.zeros(gray.shape, dtype=np.uint8)
    Vmask = np.zeros(gray.shape, dtype=np.uint8)
    MaxArea = 0
    MaxCnt = 0
    #本の領域抽出
    for i in range(len(cnt)):
        Area = cv2.contourArea(cnt[i])
        if (MaxArea<Area):
            MaxArea = Area
            MaxCnt = i
    cv2.drawContours(Amask, cnt, MaxCnt, 255, -1)
    cv2.drawContours(Cmask, cnt, MaxCnt, 255, 1)
    cv2.imwrite('../photo/other/cnt.jpg', Amask)
    #頂点の候補を調べる。
    for h in range(img.shape[0]-21):
        for w in range(img.shape[1]-21):
            j = h+11
            i = w+11
            area = 0
            AVal = Amask[j,i]
            CVal = Cmask[j,i]
            if (CVal == 255):
                for y in range(21):
                    for x in range(21):
                        if (Amask[j-10+y,i-10+x] != 0):
                            area = area+1
                if ( (21*21*0.25) <= area & area <= (21*21*0.4)):
                    Vmask[j][i] = 255
    cv2.imwrite('../photo/other/vtx.jpg', Vmask)
    #頂点を決定。
    kernel = np.ones((3, 3), np.uint8)
    Vmask = cv2.dilate(Vmask,kernel)
    vtxs, hierarchy= cv2.findContours(Vmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    Vmask = np.zeros(gray.shape, dtype=np.uint8)
    v = [[0,[0,0]],[0,[0,0]],[0,[0,0]],[0,[0,0]]]
    for i in range(len(vtxs)):
        Area = cv2.contourArea(vtxs[i])
        #モーメントから重心を求める
        M = cv2.moments(vtxs[i])
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        C = [cx,cy]
        if (v[0][0]<=Area):
            v[3] = v[2]
            v[2] = v[1]
            v[1] = v[0]
            v[0] = Area,C
        elif (v[1][0]<=Area):
            v[3] = v[2]
            v[2] = v[1]
            v[1] =Area,C
        elif (v[2][0]<=Area):
            v[3] = v[2]
            v[2] =Area,C
        elif (v[3][0]<=Area):
            v[3] =Area,C
    for i in range(4):
        Vmask[v[i][1][1]][v[i][1][0]] = 255
    kernel = np.ones((5, 5), np.uint8)
    Vmask = cv2.dilate(Vmask,kernel)
    cv2.imwrite('../photo/other/vtx2.jpg', Vmask)
    preVertexs = [v[0][1],v[1][1],v[2][1],v[3][1]]
    ver = [[0,0],[0,0],[0,0],[0,0]]
    for i in range(4):
        if (ver[0][0]<=preVertexs[i][0]):
            ver[3] = ver[2]
            ver[2] = ver[1]
            ver[1] = ver[0]
            ver[0] =preVertexs[i]
        elif (ver[1][0]<=preVertexs[i][0]):
            ver[3] = ver[2]
            ver[2] = ver[1]
            ver[1] =preVertexs[i]
        elif (ver[2][0]<=preVertexs[i][0]):
            ver[3] = ver[2]
            ver[2] =preVertexs[i]
        elif (ver[3][0]<=preVertexs[i][0]):
            ver[3] =preVertexs[i]

    if (ver[0][1] < ver[1][1]):
        c = ver[0]
        ver[0] = ver[1]
        ver[1] = c
    if (ver[3][1] < ver[2][1]):
        c = ver[3]
        ver[3] = ver[2]
        ver[2] = c

    Vertexs = np.float32(ver)
    return Vertexs

def macthing(vtx,img):
    vtxs = [vtx,np.float32([vtx[1],vtx[2],vtx[3],vtx[0]]),np.float32([vtx[2],vtx[3],vtx[0],vtx[1]]),np.float32([vtx[3],vtx[0],vtx[1],vtx[2]])]

    svtxs = np.float32([[0,0],[0,150],[150,150],[150,0]])
    M = [cv2.getPerspectiveTransform(vtxs[0],svtxs),cv2.getPerspectiveTransform(vtxs[1],svtxs),cv2.getPerspectiveTransform(vtxs[2],svtxs),cv2.getPerspectiveTransform(vtxs[3],svtxs)]

    chu = [cv2.warpPerspective(img,M[0],(150,150)),cv2.warpPerspective(img,M[1],(150,150)),cv2.warpPerspective(img,M[2],(150,150)),cv2.warpPerspective(img,M[3],(150,150))]
    gchu = [cv2.cvtColor(chu[0], cv2.COLOR_BGR2GRAY),cv2.cvtColor(chu[1], cv2.COLOR_BGR2GRAY),cv2.cvtColor(chu[2], cv2.COLOR_BGR2GRAY),cv2.cvtColor(chu[3], cv2.COLOR_BGR2GRAY)]
    ans = [0,0];
    for i in range(5):
        tmp = cv2.imread('../photo/tmp/'+str(i)+'.jpg')
        ivtxs = np.float32([[0,0],[0,tmp.shape[0]],[tmp.shape[1],tmp.shape[0]],[tmp.shape[1],0]])
        iM = cv2.getPerspectiveTransform(ivtxs,svtxs)
        tmp = cv2.warpPerspective(tmp,iM,(150,150))

        gtmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
        dimg = np.zeros(gtmp.shape, dtype=np.uint8)
        mindiff = 1000
        dmax = 0
        cv2.imwrite('../photo/other/tmp'+str(i)+'.jpg', tmp)
        for j in range(4):
            result = cv2.matchTemplate(chu[j], tmp, cv2.TM_CCOEFF_NORMED)
            min, max, min_loc, max_loc = cv2.minMaxLoc(result)
            cv2.imwrite('../photo/other/dst'+str(i)+str(j)+'.jpg', chu[j])
            diff = 0
            diff = diff/(gtmp.shape[1]*gtmp.shape[0])
            if (mindiff > diff):
                mindiff = diff
            if (dmax < max):
                dmax = max
        print(str(i)+';'+str(dmax))
        if (dmax > ans[1]):
            ans[1] = dmax
            ans[0] = i
    print('\n')
    print('---判定結果---')
    if (ans[1] > 0.3):
        if (ans[0] == 0):
            print('PHP7+MariaDB/MySQL マスターブック')
        if (ans[0] == 1):
            print('Cによるアルゴリズムとデータ構造')
        if (ans[0] == 2):
            print('未来へつなぐデジタルシリーズ24　コンパイラ')
        if (ans[0] == 3):
            print('未来へつなぐデジタルシリーズ25　オペレーティングシステム')
        if (ans[0] == 4):
            print('会社に入る前に知っておきたいこれだけ経済学')
        cv2.imshow('Ans', cv2.imread('../photo/tmp/'+str(ans[0])+'.jpg'))
        cv2.moveWindow('Ans', 700, 0)
    else :
        print('判別できませんでした')
    print('--------------')



def hosei(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
    v = hsv[:, :, 2]

    minv, maxv, min_loc, max_loc = cv2.minMaxLoc(v)

    for j in range(img.shape[0]):
        for i in range(img.shape[1]):
            hsv[j][i][2] = (v[j][i]-minv)*(255/(maxv-minv))
    return img


cap = cv2.VideoCapture(0)

time.sleep(1)
while True:
    val = cap.get(17)
    ret, frame = cap.read()
    h = frame.shape[0]
    w = frame.shape[1]

    re_frame = cv2.resize(frame,(w//3, h//3))
    cv2.imshow('Frame', re_frame)
    cv2.moveWindow('Frame', 0, 0)
    k = cv2.waitKey(50)


    if k == 27:
        break
    elif k == 13:
        cap_img = re_frame
        cap_img = hosei(re_frame)
        cv2.imwrite('../photo/other/src.jpg', cap_img)
        print('\n')
        print('判定中---')

        cnt = getCnt(cap_img)
        vtxs = getVertex(cnt,cap_img)
        macthing(vtxs,cap_img)

cap.release()
cv2.destroyAllWindows()
