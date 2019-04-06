# _*_ coding:utf-8 _*_

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

#glutSolidTeapot(size)


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


# with open('ar_camera.pkl','rb') as f:
# 	K = pickle.load(f)
# 	Rt = pickle.load(f)
# 	setup()
# 	draw_background('./data/book_perspective.bmp')
# 	set_projection_from_camera(K)
# 	set_modelview_from_camera(Rt)
# 	draw_teapot(0.05)
# 	while True:
# 		event = pygame.event.poll()
# 		if event.type in (pygame.QUIT,pygame.KEYDOWN):
# 			# break
# 			sys.exit()
# 	pygame.display.flip()

"""
while True:
	for event in pygame.event.get():
		if event.type==pygame.QUIT:
			sys.exit()
"""

