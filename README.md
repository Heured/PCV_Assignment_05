# PCV_Assignment_05
Augmented Reality test


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


2.
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
3.
```error
freeglut  ERROR:  Function <glutSolidTeapot> called without first calling 'glutInit'.
```
```python
# 在语句前加glutInit()
glutInit()
glutSolidTeapot(size)
```
