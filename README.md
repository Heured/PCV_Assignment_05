# PCV_Assignment_05
Augmented Reality test
  
## 以平面和标记物进行姿态估计
  如果图像中包含平面状的标记物体，并且已经对照相机进行了标定，那么我们可以计算出照相机的姿态(旋转和平移)。这里的标记物体可以为对任何平坦的物体。
  
  这里，我们借助以下两图提取SIFT特征，然后使用RANSAC算法稳健地估计单应性矩阵：
  
  图1：
  ![emmm](https://github.com/Heured/PCV_Assignment_05/blob/master/imgToShow/ZiTai_1.png)
  
  图2：
  ![emmm](https://github.com/Heured/PCV_Assignment_05/blob/master/imgToShow/ZiTai_2.png)
  
```python
# compute features
sift.process_image('./data/book_frontal.JPG', 'im0.sift')
l0, d0 = sift.read_features_from_file('im0.sift')

sift.process_image('./data/book_perspective.JPG', 'im1.sift')
l1, d1 = sift.read_features_from_file('im1.sift')

# match features and estimate homography
matches = sift.match_twosided(d0, d1)
ndx = matches.nonzero()[0]
fp = homography.make_homog(l0[ndx, :2].T)
ndx2 = [int(matches[i]) for i in ndx]
tp = homography.make_homog(l1[ndx2, :2].T)

model = homography.RansacModel()
H, inliers = homography.H_from_ransac(fp, tp, model)
```
  
  如此我们便得到了单应性矩阵。该单应性矩阵将一幅图像中标记物(书本)上的点映射到另一幅图像中的对应点。下面我们定义相应的三维坐标系，使标记物在X-Y平面上，原点在标记物的某个位置上。
  
  为了检验单应性矩阵结果的正确性，我们需要将一些简单的三维物体放置在标记物上，这里我们使用一个立方体。如下函数可以产生立方体上的点：
  
```python
def cube_points(c, wid):
    """ Creates a list of points for plotting
        a cube with plot. (the first 5 points are
        the bottom square, some sides repeated). """
    p = []
    # bottom
    p.append([c[0] - wid, c[1] - wid, c[2] - wid])
    p.append([c[0] - wid, c[1] + wid, c[2] - wid])
    p.append([c[0] + wid, c[1] + wid, c[2] - wid])
    p.append([c[0] + wid, c[1] - wid, c[2] - wid])
    p.append([c[0] - wid, c[1] - wid, c[2] - wid])  # same as first to close plot

    # top
    p.append([c[0] - wid, c[1] - wid, c[2] + wid])
    p.append([c[0] - wid, c[1] + wid, c[2] + wid])
    p.append([c[0] + wid, c[1] + wid, c[2] + wid])
    p.append([c[0] + wid, c[1] - wid, c[2] + wid])
    p.append([c[0] - wid, c[1] - wid, c[2] + wid])  # same as first to close plot

    # vertical sides
    p.append([c[0] - wid, c[1] - wid, c[2] + wid])
    p.append([c[0] - wid, c[1] + wid, c[2] + wid])
    p.append([c[0] - wid, c[1] + wid, c[2] - wid])
    p.append([c[0] + wid, c[1] + wid, c[2] - wid])
    p.append([c[0] + wid, c[1] + wid, c[2] + wid])
    p.append([c[0] + wid, c[1] - wid, c[2] + wid])
    p.append([c[0] + wid, c[1] - wid, c[2] - wid])

    return array(p).T
```
  
有了单应性矩阵和照相机的标定矩阵，我们现在可以得出两个视图间的相对变换：

```python

# camera calibration
K = my_calibration((747, 1000))

# 3D points at plane z=0 with sides of length 0.2
box = cube_points([0, 0, 0.1], 0.1)

# project bottom square in first image
cam1 = camera.Camera(hstack((K, dot(K, array([[0], [0], [-1]])))))
# first points are the bottom square
box_cam1 = cam1.project(homography.make_homog(box[:, :5]))

# use H to transfer points to the second image
box_trans = homography.normalize(dot(H, box_cam1))

# compute second camera matrix from cam1 and H
cam2 = camera.Camera(dot(H, cam1.P))
A = dot(linalg.inv(K), cam2.P[:, :3])
A = array([A[:, 0], A[:, 1], cross(A[:, 0], A[:, 1])]).T
cam2.P[:, :3] = dot(K, A)

# project with the second camera
box_cam2 = cam2.project(homography.make_homog(box))
```
  
再用如下代码可视化投影后的点：
```python
# plotting
im0 = array(Image.open('./data/book_frontal.JPG'))
im1 = array(Image.open('./data/book_perspective.JPG'))

figure()
imshow(im0)
plot(box_cam1[0, :], box_cam1[1, :], linewidth=3)
title('2D projection of bottom square')
axis('off')

figure()
imshow(im1)
plot(box_trans[0, :], box_trans[1, :], linewidth=3)
title('2D projection transfered with H')
axis('off')

figure()
imshow(im1)
plot(box_cam2[0, :], box_cam2[1, :], linewidth=3)
title('3D points projected in second image')
axis('off')

show()
```
  
![emmm](https://github.com/Heured/PCV_Assignment_05/blob/master/imgToShow/ZiTai_3.png)
  
  
使用Pickle将这些照相机矩阵保存成.pkl文件用于下一个例子：
```python
with open('ar_camera.pkl', 'wb') as f:
    pickle.dump(K, f)
    pickle.dump(dot(linalg.inv(K),cam2.P), f)
```
  
## 增强现实
  此例子需要安装pygame以及pyOpenGL两个工具包。
  (PS:根据经验最好还是下载.whl文件然后用pip安装，这样会免去很多麻烦)
  
使用如下函数将照相机参数转换为OpenGL中的投影矩阵：
  
  
```python

from pylab import *
import math
import pickle
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import pygame, pygame.image
from pygame.locals import *
from PCV.geometry import homography, camera
from PCV.localdescriptors import sift


"""
This is the augmented reality and pose estimation cube example from Section 4.3.
"""

def set_projection_from_camera(K):
	glMatrixMode(GL_PROJECTION)
	glLoadIdentity()
	fx = K[0,0]
	fy = K[1,1]
	fovy = 2*math.atan(0.5*height/fy)*180/math.pi
	aspect = (width*fy)/(height*fx)
	near = 0.1
	far = 100.0
	gluPerspective(fovy,aspect,near,far)
	glViewport(0,0,width,height)

```
  
再使用下面的函数实现获得移除标定矩阵后的3*4针孔照相机矩阵，同时创建一个模拟视图：
  
```python

def set_modelview_from_camera(Rt):
	glMatrixMode(GL_MODELVIEW)
	glLoadIdentity()
	Rx = np.array([[1,0,0],[0,0,-1],[0,1,0]])
	R = Rt[:,:3]
	U,S,V = np.linalg.svd(R)
	R = np.dot(U,V)
	R[0,:] = -R[0,:]
	t = Rt[:,3]
	M = np.eye(4)
	M[:3,:3] = np.dot(R,Rx)
	M[:3,3] = t
	M = M.T
	m = M.flatten()
	glLoadMatrixf(m)
```
  
使用下列函数载入一个图像，将其转换成OpenGL纹理，并将该纹理放置在四边形上：
  
```python

def draw_background(imname):
	bg_image = pygame.image.load(imname).convert()
	bg_data = pygame.image.tostring(bg_image,"RGBX",1)
	glMatrixMode(GL_MODELVIEW)
	glLoadIdentity()

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
	glEnable(GL_TEXTURE_2D)
	glBindTexture(GL_TEXTURE_2D,glGenTextures(1))
	glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,width,height,0,GL_RGBA,GL_UNSIGNED_BYTE,bg_data)
	glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST)
	glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST)
	glBegin(GL_QUADS)
	glTexCoord2f(0.0,0.0); glVertex3f(-1.0,-1.0,-1.0)
	glTexCoord2f(1.0,0.0); glVertex3f( 1.0,-1.0,-1.0)
	glTexCoord2f(1.0,1.0); glVertex3f( 1.0, 1.0,-1.0)
	glTexCoord2f(0.0,1.0); glVertex3f(-1.0, 1.0,-1.0)
	glEnd()
	glDeleteTextures(1)

```
  
用下面的函数给茶壶上色：
  
```python

def draw_teapot(size):

	glEnable(GL_LIGHTING)
	glEnable(GL_LIGHT0)
	glEnable(GL_DEPTH_TEST)
	glClear(GL_DEPTH_BUFFER_BIT)
	glMaterialfv(GL_FRONT,GL_AMBIENT,[0,0,0,0])
	glMaterialfv(GL_FRONT,GL_DIFFUSE,[0.5,0.0,0.0,0.0])
	glMaterialfv(GL_FRONT,GL_SPECULAR,[0.7,0.6,0.6,0.0])
	glMaterialf(GL_FRONT,GL_SHININESS,0.25*128.0)
	# glutInit()
	# glutWireTeapot(size)
	glutSolidTeapot(size)
```
  
再进行如下脚本可以生成结果图：
  
```python


width,height = 1000,747

def setup():
	pygame.init()
	pygame.display.set_mode((width,height),OPENGL | DOUBLEBUF)
	pygame.display.set_caption("OpenGL AR demo")



# 载入照相机数据

f = open('ar_camera.pkl','rb')
K = pickle.load(f)
Rt = pickle.load(f)

setup()
draw_background('./data/book_perspective.bmp')
set_projection_from_camera(K)
set_modelview_from_camera(Rt)

draw_teapot(0.05)

pygame.display.flip()
print(K)
print(Rt)
while True:
	for event in pygame.event.get():
		if event.type==pygame.QUIT:
			sys.exit()

```
  
结果图：
  
![emmmm](https://github.com/Heured/PCV_Assignment_05/blob/master/imgToShow/theTeapot.PNG)
  

  
遇到问题：
1.
```error
Traceback (most recent call last):
  File "D:/pyCharm/pycharm_workspace/2019-4-2ZTGuJi/ZiTaiGuJi.py", line 124, in <module>
    pickle.dump(K, f)
TypeError: write() argument must be str, not bytes
```
百度之后说改成wb可以解决
```python

#with open('ar_camera.pkl','w') as f:
with open('ar_camera.pkl','wb') as f:
```
  
  
```error
Traceback (most recent call last):
  File "D:/pyCharm/pycharm_workspace/2019-4-2ZTGuJi/Prepare.py", line 132, in <module>
    K = pickle.load(f)
UnicodeDecodeError: 'gbk' codec can't decode byte 0x80 in position 0: illegal multibyte sequence
```
因为之前生成pkl文件因为报错把w改成了wb，所以这里的r也要改成rb
```python

#with open('ar_camera.pkl','r') as f:
with open('ar_camera.pkl','rb') as f:
```
  
  
2.
```error
freeglut  ERROR:  Function <glutSolidTeapot> called without first calling 'glutInit'.
```
```python
# 在语句前加glutInit()
glutInit()
glutSolidTeapot(size)
```
  
然后出现了这个..
```error
Traceback (most recent call last):
  File "D:/pyCharm/pycharm_workspace/2019-4-2ZTGuJi/main.py", line 153, in <module>
    draw_teapot(0.05)
  File "D:/pyCharm/pycharm_workspace/2019-4-2ZTGuJi/main.py", line 111, in draw_teapot
    glutSolidTeapot(size)
OSError: exception: access violation reading 0x00000000000000C1
```
emmmm这..
  
  
  
解决办法：
  
![emmmm](https://github.com/Heured/PCV_Assignment_05/blob/master/imgToShow/solution_to_freeglut.PNG)

