# import cv2
# #读取图片,默认是将图像调整为3通道的BGR图像,打印读取的图像数据
# std=cv2.imread('./image/std2.png')
# print(std)
# # #创建一个名为sjj的窗口
# # cv2.namedWindow('lesson')
# # #imshow函数用于在窗口sjj内,显示图像
# # cv2.imshow('lesson',std)
# # cv2.waitKey()
# #可以先不用创建窗口
# cv2.imshow('demo',std)

# import cv2
# std=cv2.imread("./image/std2.png")
# cv2.imshow("demo",std)
# key=cv2.waitKey()
# if key==ord('A'):
#     cv2.imshow("PressA",std)
# elif key==ord('B'):
#     cv2.imshow("PressB",std)

# import cv2
# std=cv2.imread("./image/std2.png")
# cv2.imshow("demo",std)
# key=cv2.waitKey()#等待用户按下按键
# if key!=-1:
#     print('用户按下了按键')

# import cv2
# std=cv2.imread("./image/std2.png")
# cv2.imshow("demo1",std)
# cv2.imshow("demo2",std)
# cv2.waitKey()
# #一下子释放所有的窗口
# cv2.destroyAllWindows()
# #释放指定窗口
# cv2.destroyWindow("demo")

# import cv2
# std=cv2.imread("./image/std.jpg")
# r=cv2.imwrite('./image/result.bmp',std)


# #二值图像以及灰度图像,二维数组与图像之间存在对应关系
# import cv2
# import numpy as np
# #8行8列的矩阵模拟纯黑的图片
# img=np.zeros((8,8),dtype=np.uint8)
# print("img=\n",img)
# cv2.imshow("one",img)
# print('读取像素点img[0,3]=',img[0,3])
# img[0,3]=255
# print('修改后img=\n',img)
# print('读取修改后的图像img[0,3]=',img[0,3])
# cv2.imshow("two",img)
# # while(True):
# #     pass
# cv2.waitKey()
# cv2.destroyAllWindows()


#实战:读取一幅灰度图像,并对其像素进行访问修改
# import cv2
# img=cv2.imread('./image/test.jpg',0)
# cv2.imshow('before',img)
# for i in range(10,100):
#     for j in range(80,100):
#         img[i,j]=255
# cv2.imshow('after',img)
# #这样写只是为了方便截图
# i=0
# while(i<4):
#  cv2.waitKey()
#  i=i+1
#
# cv2.destroyAllWindows()


#彩色图像处理,RGB彩色图像读入时按照行方向依次BGR通道,以行为单位
#存储在ndarray的列中
#img[0,0,0]表示访问图像image的B通道内的第0行第0列上的像素点
#式中第1个索引表示第0行,第2个索引表示第0列,第3个索引表示第0个颜色通道
#在opencv中通道读取的顺序依次是B,G,R
#使用Numpy库来生成一个2*4*3大小的数组,用它模拟一幅黑色图像
# import numpy as np
# import cv2
# #------蓝色通道-------
# blue=np.zeros((300,300,3),dtype=np.uint8)
# #针对数组blue,将其第0个通道的值设置为255,图像blue的值为255
# blue[:,:,0]=255
# print('blue=\n',blue)
# cv2.imshow('blue',blue)
# #------绿色通道-------
# green=np.zeros((300,300,3),dtype=np.uint8)
# green[:,:,1]=255
# print('green=\n',green)
# cv2.imshow('green',green)
# #------红色通道--------
# red=np.zeros((300,300,3),dtype=np.uint8)
# red[:,:,2]=255
# print('red=\n',red)
# cv2.imshow('red',red)
# #------释放窗口--------
# i=0
# while i<4:
#  cv2.waitKey()
#  i+=1
# cv2.destroyAllWindows()


# #使用numpy生成一个三维数组,用来观察三个通道值的变化情况
# import numpy as np
# import cv2
# img=np.zeros((300,300,3),dtype=np.uint8)
# img[:,100:200,0]=255
# img[:,200:300,1]=255
# img[:,0:100,2]=255
# print('img=\n',img)
# cv2.imshow('img',img)
# cv2.waitKey()
# cv2.destroyAllWindows()


# #使用Numpy生成一个三维数组,用来模拟一幅BGR模式的彩色图像,并对其进行修改,访问
# import numpy as np
# import cv2
# img=np.zeros((2,4,3),dtype=np.uint8)
# print("img=\n",img)
# print('读取像素点img[0,3]=',img[0,3])
# print('读取像素点img[1,2,2]=',img[1,2,2])
# #访问第0行第3列上的R,G,B通道的三个像素点
# img[0,3]=255
# img[0,0]=[66,77,88]
# img[1,1,1]=3
# img[1,2,2]=4
# img[0,2,0]=5
# print('修改后img\n',img)
# print('读取修改后像素点img[1,2,2]=',img[1,2,2])
# # cv2.imshow('img',img)
# # cv2.waitKey()
# # cv2.destroyAllWindows()


# #读取一幅彩色图像,并对其像素进行访问,修改
# import numpy as np
# import cv2
# img=cv2.imread('./image/iu.jpeg',1)
# cv2.imshow('before',img)
# print('访问img[0,0]=',img[0,0])#输出第0行第0列的BGR的取值
# print('访问img[0,0,0]=',img[0,0,0])#输出图片的第0行第0列的B通道的值
# print('访问img[0,0,1]=',img[0,0,1])
# print('访问img[0,0,2]=',img[0,0,2])
# print('访问img[50,0]=',img[50,0])
# print('访问img[100,0]=',img[100,0])
# #区域1
# for i in range(0,50):
#     for j in range(0,100):
#         for k in range(0,3):
#             img[i,j,k]=255
# #区域2
# for i in range(50,100):
#     for j in range(0,100):
#         img[i,j]=[128,128,128]
# #区域3
# for i in range(100,150):
#     for j in range(0,100):
#         img[i,j]=0
# cv2.imshow('after',img)
# print('访问img[0,0]=',img[0,0])#输出第0行第0列的BGR的取值
# print('访问img[0,0,0]=',img[0,0,0])#输出图片的第0行第0列的B通道的值
# print('访问img[0,0,1]=',img[0,0,1])
# print('访问img[0,0,2]=',img[0,0,2])
# print('访问img[50,0]=',img[50,0])
# print('访问img[100,0]=',img[100,0])
# i=0
# while i<4:
#     i+=1
#     cv2.waitKey()
#
# cv2.destroyAllWindows()


#使用numpy.array()访问像素
#numpy.array()提供了item()和itemset()来访问和修改函数
#使用numpy.array()比直接使用索引更快
#使用Numpy生成一个二维随机数组,用来模拟一幅灰度图像,并对其像素进行访问,修改
# import numpy as np
# img=np.random.randint(10,99,size=[5,5],dtype=np.uint8)
# print('img=\n',img)
# print('读取像素点img.item(3,2)=',img.item(3,2))
# img.itemset((3,2),255)
# print('修改后 img=\n',img)
# print('修改后像素点img.item(3,2)=',img.item(3,2))


# #生成一个灰度图像,让其中的像素值均为随机数
# import numpy as np
# import cv2
# img=np.random.randint(0,256,size=[256,256],dtype=np.uint8)
# cv2.imshow('demo',img)
# i=0
# while i<4:
#     i+=1
#     cv2.waitKey()
#
# cv2.destroyAllWindows()

#读取一幅灰度图像,并对其像素值进行访问,修改
# import cv2
# img=cv2.imread('./image/test.jpg',0)
# print(img)
# #测试读取,修改单个像素值
# print('读取像素点img.item(3,2)=',img.item(3,2))
# img.itemset((3,2),255)
# print('修改后像素点img.item(3,2)=',img.item(3,2))
# #测试修改一个区域的像素值
# cv2.imshow('before',img)
# for i in range(10,100):
#     for j in range(80,100):
#         img.itemset((i,j),255)
# cv2.imshow('after',img)
# i=0
# while i<4:
#     i+=1
#     cv2.waitKey()
# cv2.destroyAllWindows()


# #使用Numpy生成一个有=由随机数构成的三维数组,用来模拟一幅RGB色彩空间
# #的彩色图像,并使用函数item()和itemset()来访问和修改它
# import numpy as np
# img=np.random.randint(10,99,size=[2,4,3],dtype=np.uint8)
# print('img=\n',img)
# print('读取像素点img[1,2,0]=',img.item(1,2,0))
# print('读取像素点img[0,2,1]=',img.item(0,2,1))
# print('读取像素点img[1,0,2]=',img.item(1,0,2))
# img.itemset((1,2,0),255)
# img.itemset((0,2,1),255)
# img.itemset((1,0,2),255)
# print('修改后img=\n',img)
# print('修改后像素点img[1,2,0]=',img.item(1,2,0))
# print('修改后像素点img[0,2,1]=',img.item(0,2,1))
# print('修改后像素点img[1,0,2]=',img.item(1,0,2))


# #生成一幅彩色图像,让其中的像素值均为随机数
# import cv2
# import numpy as np
# img=np.random.randint(0,256,size=[256,256,3],dtype=np.uint8)
# cv2.imshow('demo',img)
# i=0
# while i<4:
#     i+=1
#     cv2.waitKey()
# cv2.destroyAllWindows()


#读取一幅彩色图像,并对其像素进行访问,修改
# import cv2
# import numpy as np
# img=cv2.imread('./image/iu.jpeg',1)
# cv2.imshow('before',img)
# print('访问前img[0,0,0]=',img.item(0,0,0))
# print('访问前img[0,0,1]=',img.item(0,0,1))
# print('访问前img[0,0,2]=',img.item(0,0,2))
# for i in range(0,50):
#     for j in range(0,100):
#         for k in range(0,3):
#             img.itemset((i,j,k),255)#白色
# cv2.imshow('after',img)
# print('修改后img[0,0,0]=',img.item(0,0,0))
# print('修改后img[0,0,1]=',img.item(0,0,1))
# print('修改后img[0,0,2]=',img.item(0,0,2))
# i=0
# while i<4:
#     i+=1
#     cv2.waitKey()
# cv2.destroyAllWindows()

#
# #感兴趣区域(ROI)
# import cv2
# a=cv2.imread('./image/iu.jpeg',-1)
# face=a[220:400,150:350]
# cv2.imshow("face",face)
# i=0
# while i<4:
#     i+=1
#     cv2.waitKey()
# cv2.destroyAllWindows()


# #对图像中的人物的脸部进行打码
# import cv2
# import numpy as np
# a=cv2.imread('./image/iu.jpeg',-1)
# cv2.imshow('original',a)
# face=np.random.randint(0,256,(180,200,3))
# print(face)
# a[220:400,150:350]=face
# cv2.imshow('result',a)
# i=0
# while i<4:
#     i+=1
#     cv2.waitKey()
# cv2.destroyAllWindows()


# #将一幅图像的ROI复制到另一幅图像内
# import cv2
# iu=cv2.imread('./image/iu.jpeg',-1)
# ym=cv2.imread('./image/ym.jpeg',-1)
# cv2.imshow('iu',iu)
# cv2.imshow('ym',ym)
# face=iu[200:450,150:300]
# # ymf=ym[50:300,200:450]
# ym[50:300,300:450]=face
# # cv2.imshow('ymf',ymf)
# cv2.imshow('result',ym)
# i=0
# while i<4:
#     i+=1
#     cv2.waitKey()
# cv2.destroyAllWindows()

#演示图像通道拆分及通道值改变对彩色图像的影响
# import cv2
# iu=cv2.imread('./image/iu.jpeg',-1)
# cv2.imshow('iu',iu)
# #通过索引拆分通道
# b=iu[:,:,0]#b通道
# g=iu[:,:,1]#g通道
# r=iu[:,:,2]#r通道
# cv2.imshow("b",b)
# cv2.imshow("g",g)
# cv2.imshow("r",r)
# iu[:,:,0]=0#b通道全部设置为0
# cv2.imshow('iu0',iu)
# iu[:,:,1]=0#b,g通道全部设置为0
# cv2.imshow('iu1',iu)
# iu[:,:,2]=0
# cv2.imshow('iu2',iu)
# i=0
# while i<4:
#     i+=1
#     cv2.waitKey()
# cv2.destroyAllWindows()


#通过函数拆分
#b,g,r=cv2.split(img)
#与b=cv2.split(a)[0]
#g=cv2.split(a)[1]
#r=cv2.split(a)[2]等价
# import cv2
# iu=cv2.imread('./image/iu.jpeg',-1)
# b,g,r=cv2.split(iu)
# cv2.imshow('B',b)
# cv2.imshow('G',g)
# cv2.imshow('R',r);i=0
# while i<4:
#     i+=1
#     cv2.waitKey()
# cv2.destroyAllWindows()


# import cv2
# iu=cv2.imread('./image/iu.jpeg',-1)
# b,g,r=cv2.split(iu)
# bgr=cv2.merge([b,g,r])
# rgb=cv2.merge([r,g,b])
# cv2.imshow('iu',iu)
# cv2.imshow('bgr',bgr)
# cv2.imshow('rgb',rgb);i=0
# while i<4:
#     i+=1
#     cv2.waitKey()
# cv2.destroyAllWindows()

#观察图像的常用属性
# import cv2
# gray=cv2.imread('./image/test.jpg',0)
# color=cv2.imread('./image/iu.jpeg',1)
# print('图像gray的属性是: ')
# #shape:彩色图像则返回行,列,通道的数组;二值或灰度图像返回行数和列数
# #size:返回图像的像素数目.行*列*通道数,灰二的通道数是1
# #dtype:返回图像的数据类型
# print('gray.shape=',gray.shape)
# print('gray.size=',gray.size)
# print('gray.dtype=',gray.dtype)
# print('图像color的属性是: ')
# print('color.shape=',color.shape)
# print('color.size=',color.size)
# print('color.dtype=',color.dtype)


#使用随机数数组模拟灰度图像,观察使用'+'对像素值求和的结果
# import numpy as np
# img1=np.random.randint(0,256,size=[3,3],dtype=np.uint8)
# img2=np.random.randint(0,256,size=[3,3],dtype=np.uint8)
# print('img1=\n',img1)
# print('img2=\n',img2)
# print('img1+img2=\n',img1+img2)


# #使用随机数数组模拟灰度图像,观察函数cv2.add()对像素值求和的结果
# import numpy as np
# import cv2
# img1=np.random.randint(0,256,size=[3,3],dtype=np.uint8)
# img2=np.random.randint(0,256,size=[3,3],dtype=np.uint8)
# print('img1=\n',img1)
# print('img2=\n',img2)
# img3=cv2.add(img1,img2)
# print('img1+img2=\n',img3)


# #分别使用加号运算符和函数cv2.add()计算两幅灰度图像的像素值之和
# import cv2
# a=cv2.imread('./image/test.jpg',0)
# b=a
# result1=a+b
# result2=cv2.add(a,b)
# cv2.imshow('original',a)
# cv2.imshow('result1',result1)
# cv2.imshow('result2',result2)
# # cv2.waitKey()
# i=0
# while i<4:
#     i+=1
#     cv2.waitKey()
# cv2.destroyAllWindows()


#图像加权和
#计算两幅图像的像素值之和的时候,考虑权重,dst=saturate(src1*alpha+src2*beta+gamma)
#要求src1和src2必须大小,类型相同,opencv中提供了函数cv2.addWeighted()实现图像的加权和
#dst=cv2.addWeighted(src1,alpha,src2,beta,gamma)
#可以理解为结果图像=图像1*系数1+图像2*系数2+亮度调节量
#使用数组演示函数cv2.addWeighted()的使用
# import cv2
# import numpy as np
# img1=np.ones((3,4),dtype=np.uint8)*100
# img2=np.ones((3,4),dtype=np.uint8)*10
# gamma=3
# img3=cv2.addWeighted(img1,0.6,img2,5,gamma)
# print(img3)


# #使用函数cv2.addWeighted()函数对两幅图进行加权混合,观察处理结果
# import cv2
# #如果不添加后面的-1,则默认输出是三通道的
# iu=cv2.imread('./image/test.jpg',0)
# ym=cv2.imread('./image/test3.jpeg',0)
# print(iu,'\n',ym)
# print(iu.size,' ',ym.size)
# #src1,src2必须大小,类型相同,否则会报错哦
# result=cv2.addWeighted(iu,0.6,ym,0.4,0)
# cv2.imshow('iu',iu)
# cv2.imshow('ym',ym)
# cv2.imshow('result',result)
# i=0
# while i<4:
#     i+=1
#     cv2.waitKey()
# cv2.destroyAllWindows()

# import cv2
# iu=cv2.imread('./image/iu.jpeg',0)
# cv2.imshow('iu',iu)
# cv2.waitKey()
# cv2.destroyAllWindows()


#使用函数cv2.addWeighted()将一幅图的ROI混合在另一幅图像内
# import cv2
# test=cv2.imread('./image/test.jpg',-1)
# test3=cv2.imread('./image/test3.jpeg',-1)
# cv2.imshow('test',test)
# cv2.imshow('test3',test3)
# #记住提取的大小一定要相同
# face1=test[220:400,250:350]
# face2=test3[300:480,300:400]
# add=cv2.addWeighted(face1,0.6,face2,0.4,0)
# test3[300:480,300:400]=add
# cv2.imshow('result',test3)
# i=0
# while i<4:
#     i+=1
#     cv2.waitKey()
# cv2.destroyAllWindows()
# cv2.destroyAllWindows()


#在opencv中,可以使用cv2.bitwise_and()来实现按位与运算
#dst=cv2.bitwise_and(src1,src2[,mask]]),mask表示可选操作掩码,8位单通道array
#使用数组演示与掩膜图像的按位与操作
# import cv2
# import numpy as np
# a=np.random.randint(0,255,size=[5,5],dtype=np.uint8)
# b=np.zeros((5,5),dtype=np.uint8)
# b[0:3,0:3]=255
# b[4,4]=255
# c=cv2.bitwise_and(a,b)
# print('a=\n',a)
# print('b=\n',b)
# print('c=\n',c)


# #构造一个掩膜图像,使用按位与运算保留图像中被掩膜指定的部分
# import cv2
# import numpy as np
# a=cv2.imread('./image/test.jpg',0)
# b=np.zeros(a.shape,dtype=np.uint8)
# b[100:400,200:400]=255
# b[100:500,100:200]=255
# c=cv2.bitwise_and(a,b)
# cv2.imshow('a',a)
# cv2.imshow('b',b)
# cv2.imshow('c',c)
# i=0
# while i<4:
#     i+=1
#     cv2.waitKey()
# cv2.destroyAllWindows()


#有时候我们还需要对BGR模式的彩色图像进行掩膜提取指定部分,
#由于按位于要求参与运算的数据有相同的通道,所以,需要将掩膜图像转换为BGR模式的彩色图像.
#让彩色图像与掩膜图像进行按位与操作
# import cv2
# import numpy as np
# a=cv2.imread('./image/test.jpg',1)
# b=np.zeros(a.shape,dtype=np.uint8)
# b[100:400,200:400]=255
# b[100:500,100:200]=255
# c=cv2.bitwise_and(a,b)
# print('a.shape=',a.shape)
# print('b.shape=',b.shape)
# cv2.imshow('a',a)
# cv2.imshow('b',b)
# cv2.imshow('c',c)
# i=0
# while i<4:
#     i+=1
#     cv2.waitKey()
# cv2.destroyAllWindows()


#按位或运算
#dst=cv2.bitwise_or(src1,src2[,mask]])
#按位非运算
#dst=cv2.bitwise_not(src[,mask])
#按位异或运算
#dst=cv2.bitwise_xor(src1,src2[,mask]])

#掩模
#opencv中的很多函数都会指定一个掩摸,也被称为掩码
#计算结果=cv2.add(参数1,参数2,掩模)
#当使用掩模时,操作只会在掩模值为非空的像素点上执行,并将其他像素点的值置为0
#演示掩码的作用
# import cv2
# import numpy as np
# img1=np.ones((4,4),dtype=np.uint8)*3
# img2=np.ones((4,4),dtype=np.uint8)*5
# mask=np.zeros((4,4),dtype=np.uint8)
# mask[2:4,2:4]=1
# img3=np.ones((4,4),dtype=np.uint8)*66
# print('img1=\n',img1)
# print('img2=\n',img2)
# print('mask=\n',mask)
# print('初始值img3=\n',img3)
# img3=cv2.add(img1,img2,mask=mask)
# print('求和后img3=\n',img3)



#构造一个掩模图像,将该掩模图像作为按位与函数的掩模参数,实现保留图像的指定部分
# import cv2
# import numpy as np
# a=cv2.imread('./image/test.jpg',1)
# w,h,c=a.shape
# mask=np.zeros((w,h),dtype=np.uint8)
# mask[100:400,200:400]=255
# mask[100:500,100:200]=255
# c=cv2.bitwise_and(a,a,mask=mask)
# print('a.shape=',a.shape)
# print('mask.shape=',mask.shape)
# cv2.imshow('a',a)
# cv2.imshow('mask',mask)
# cv2.imshow('c',c)
# i=0
# while i<4:
#     i+=1
#     cv2.waitKey()
# cv2.destroyAllWindows()


#演示图像与数值的运算结果
# import cv2
# import numpy as np
# img1=np.ones((4,4),dtype=np.uint8)*3
# img2=np.ones((4,4),dtype=np.uint8)*5
# print('img1=\n',img1)
# print('img2=\n',img2)
# img3=cv2.add(img1,img2)
# print('cv2.add(img1,img2)=\n',img3)
# img4=cv2.add(img1,6)
# print('cv2.add(img1,6)\n',img4)
# img5=cv2.add(6,img2)
# print('cv2.add(6,img2)=\n',img5)


#位平面分解
#将灰度图像中处于同一比特位上的二进制像素值进行组合,得到一幅二进值图像,该图像即灰度图像的一个
#位平面,该过程称为位平面分解
#平面分解的具体步骤:
#1.图像预处理:读取原始图像,并且获取宽度和高度
#2.构造提取矩阵:建立一个值均为2^n的Mat作为提取矩阵(数组),用来与原始图像
#进行按位与运算,提取第n个位平面
#3.提取位平面,将位平面与提取矩阵进行按位运算得到各个位平面
#4.阈值处理.若直接显示得到的位平面,则会得到一张近似黑色的图像
#每次提取位平面后,要想让二值位平面能够以黑白颜色显示出来,就要进行阈值处理,将大于0的值变为255
#5.显示图像
#完成上述处理后,可以将位平面显示出来,直观地观察各个位平面的具体情况
#编写程序,观察灰度图像的各个位平面
# import cv2
# import numpy as np
# iu=cv2.imread('./image/iu.jpeg',0)
# cv2.imshow('iu',iu)
# r,c=iu.shape
# x=np.zeros((r,c,8),dtype=np.uint8)
# #用于提取各个位平面的提取矩阵
# for i in range(8):
#     x[:,:,i]=2**i
#
# r=np.zeros((r,c,8),dtype=np.uint8)
# #实现各个位平面的提取,阈值处理,显示
# for i in range(8):
#     r[:,:,i]=cv2.bitwise_and(iu,x[:,:,i])
#     mask=r[:,:,i]>0
#     r[mask]=255
#     cv2.imshow(str(i),r[:,:,i])
# i=0
# while i<4:
#     i+=1
#     cv2.waitKey()
# cv2.destroyAllWindows()


#图像加密,解密
#通过按位异或实现图像的加密和解密
#原始图像和密钥图像进行异实现加密;加密后的图像和密钥图像进行异或,可以实现解密
#xor(a,b)=c
#xor(a,c)=b;xor(b,c)=a
#编写程序,通过图像按位异或实现加密和解密过程
#在本例中将随机生成一幅图像作为密钥
# import cv2
# import numpy as np
# iu=cv2.imread('./image/iu.jpeg',0)
# r,c=iu.shape
# key=np.random.randint(0,256,size=[r,c],dtype=np.uint8)
# encryption=cv2.bitwise_xor(iu,key)
# decryption=cv2.bitwise_xor(encryption,key)
# cv2.imshow('iu',iu)
# cv2.imshow('key',key)
# cv2.imshow('encryption',encryption)
# cv2.imshow('decryption',decryption)
# i=0
# while i<4:
#     i+=1
#     cv2.waitKey()
# cv2.destroyAllWindows()



#编写程序,实现数字水印的提取和嵌入过程
#嵌入过程:初始化-->载体图像预处理-->建立提取矩阵-->载体图像最低有效位置为0
#-->水印图像处理-->嵌入水印-->显示原始图像,水印,含水印图像
#提取过程:初始化-->含水印图像处理-->建立提取矩阵-->提取水印信息-->计算删除水印后的载体图像
#-->显示图像
# import cv2
# import numpy as np
# #读取原始载体图像
# iu=cv2.imread('./image/iu.jpeg',0)
# print(iu.shape)
# #读取水印图像
# watermark=cv2.imread('./image/wm1.jpeg',0)
# print(watermark)
# print(watermark.shape)
# #将水印图像内的值255处理为1,以方便嵌入
# #后续章节会介绍threshold处理
# w=watermark[:,:]>0
# watermark[w]=1
# #读取原始载体图像的shape值
# r,c=iu.shape
# #==============嵌入过程================
# #生成元素值都是254的数组
# t254=np.ones((r,c),dtype=np.uint8)*254
# #获取iu图像的高7位
# iuH7=cv2.bitwise_and(iu,t254)
# #将watermark嵌入iuH7图像内
# e=cv2.bitwise_or(iuH7,watermark)
# #==============提取过程================
# #生成元素值都是1的数组
# t1=np.ones((r,c),dtype=np.uint8)
# #从载体图像内提取水印图像
# wm=cv2.bitwise_and(e,t1)
# print(wm)
# #将水印图像内的值1处理为255,以方便显示
# #后续章节会介绍使用threshold实现
# w=wm[:,:]>0
# wm[w]=255
# #===============显示==================
# cv2.imshow('iu',iu)
# cv2.imshow('watermark',watermark*255)
# cv2.imshow('e',e)
# cv2.imshow('wm',wm)
# i=0
# while i<4:
#     i+=1
#     cv2.waitKey()
# cv2.destroyAllWindows()



#编写程序使用掩膜对iu图像的脸部进行打码和解码
# import cv2
# import numpy as np
# #读取原始载体图像
# iu=cv2.imread('./image/iu.jpeg',0)
# #读取原始载体图像的shape值
# r,c=iu.shape
# mask=np.zeros((r,c),dtype=np.uint8)
# mask[220:400,150:350]=1
# #获取一个key,打码,解码时所需要的密钥
# key=np.random.randint(0,256,size=[r,c],dtype=np.uint8)
# #=================获取打码脸==================
# #使用密钥key对原始图像iu进行加密
# iuXORkey=cv2.bitwise_xor(iu,key)
# #获取加密图像的脸部信息encryptFace
# encryptFace=cv2.bitwise_and(iuXORkey,mask*255)
# #将图像iu的脸部值设为0,得到noFace1
# noFace1=cv2.bitwise_and(iu,(1-mask)*255)
# #得到打码的iu图像
# maskFace=encryptFace+noFace1
# #================将打码脸解码================
# #将脸部打码的iu与密钥key进行异或运算,得到脸部的原始信息
# extractOriginal=cv2.bitwise_xor(maskFace,key)
# #将解码的脸部信息extractOriginal提取出来,得到脸部的原始信息
# extractFace=cv2.bitwise_and(extractOriginal,mask*255)
# #从脸部打码的iu内提取没有脸部信息的iu图像,得到noFace2
# noFace2=cv2.bitwise_and(maskFace,(1-mask)*255)
# #得到解码的iu图像
# extractIu=noFace2+extractFace
# #=================显示图像==================
# cv2.imshow('iu',iu)
# cv2.imshow('mask',mask*255)
# cv2.imshow('1-mask',(1-mask)*255)
# cv2.imshow('key',key)
# cv2.imshow('iuXorKey',iuXORkey)
# cv2.imshow('encryptFace',encryptFace)
# cv2.imshow('noFace1',noFace1)
# cv2.imshow('maskFace',maskFace)
# cv2.imshow('extractOriginal',extractOriginal)
# cv2.imshow('extractFace',extractFace)
# cv2.imshow('noFace2',noFace2)
# cv2.imshow('extractIu',extractIu)
# i=0
# while i<6:
#     i+=1
#     cv2.waitKey()
# cv2.destroyAllWindows()


#使用cv2.cvtColor()的实例来观察色彩空间转换功能
#将BGR图像转换为灰度图像
# import cv2
# import numpy as np
# img=np.random.randint(0,256,size=[2,4,3],dtype=np.uint8)
# rst=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# print('img=\n',img)
# print('rst=\n',rst)
# print('像素点(1,0)直接计算得到的值=',img[1,0,0]*0.114+img[1,0,1]*0.587+img[1,0,2]*0.299)
# print('像素点(1,0)使用公式cv2.cvtColor()转换值=',rst[1,0])


# #将灰度图像转换为BGR图像
# import cv2
# import numpy as np
# img=np.random.randint(0,256,size=[2,4],dtype=np.uint8)
# rst=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
# print('img=\n',img)
# print('rst=\n',rst)


#将图像在RGB和BGR模式之间转换
# import cv2
# import numpy as np
# img=np.random.randint(0,256,size=[2,4,3],dtype=np.uint8)
# rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
# print('img=\n',img)
# print('rgb=\n',rgb)
# print('bgr=\n',bgr)


#使用函数cv2.cvtcolor()来处理图像类型的转换
#将图像在BGR模式和灰度图像之间转换
# import cv2
# iu=cv2.imread('./image/iu.jpeg')
# gray=cv2.cvtColor(iu,cv2.COLOR_BGR2GRAY)
# bgr=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)#因为由灰度图像的转换3个通道值一样所以仍是灰度图像
# #=============打印shape=================
# print('iu.shape=',iu.shape)
# print('gray.shape=',gray.shape)
# print('bgr.shape=',bgr.shape)
# #==============显示效果==================
# cv2.imshow('iu',iu)
# cv2.imshow('gray',gray)
# cv2.imshow('bgr',bgr)
# cv2.waitKey()
# cv2.destroyAllWindows()


#将图像从BGR模式转换为RGB模式
# import cv2
# iu=cv2.imread('./image/iu.jpeg')
# rgb=cv2.cvtColor(iu,cv2.COLOR_BGR2RGB)
# cv2.imshow('iu',iu)
# cv2.imshow('rgb',rgb)
# cv2.waitKey()
# cv2.destroyAllWindows()


#在opencv中,测试RGB色彩空间中不同颜色的值转换到HSV色彩空间后的对应值
# import cv2
# import numpy as np
# #============测试一下Opencv中蓝色的HSV模式值==================
# imgBlue=np.zeros([1,1,3],dtype=np.uint8)
# imgBlue[0,0,0]=255
# Blue=imgBlue
# BlueHSV=cv2.cvtColor(Blue,cv2.COLOR_BGR2HSV)
# print('Blue=\n',Blue)
# print('BlueHSV=\n',BlueHSV)
# #============测试一下Opencv中绿色的HSV模式值==================
# imgGreen=np.zeros([1,1,3],dtype=np.uint8)
# imgGreen[0,0,1]=255
# Green=imgGreen
# GreenHSV=cv2.cvtColor(Green,cv2.COLOR_BGR2HSV)
# print('Green=\n',Green)
# print('GreenHSV=\n',GreenHSV)
# #============测试一下OpenCV中红色的HSV模式值=================
# imgRed=np.zeros([1,1,3],dtype=np.uint8)
# imgRed[0,0,2]=255
# Red=imgRed
# RedHSV=cv2.cvtColor(Red,cv2.COLOR_BGR2HSV)
# print('Red=\n',Red)
# print('RedHSV=\n',RedHSV)

#使用函数cv2.inRange()将某个图像内的在[100,200]内的值标注出来
# import cv2
# import numpy as np
# img=np.random.randint(0,256,size=[5,5],dtype=np.uint8)
# min=100
# max=200
# mask=cv2.inRange(img,min,max)
# print('img=\n',img)
# print('mask=\n',mask)


#通过基于掩码的按位与显示ROI
#正常显示某个图像内的感兴趣区域(ROI),而将其余区域显示为黑色
# import cv2
# import numpy as np
# img=np.ones([5,5],dtype=np.uint8)*9
# mask=np.zeros([5,5],dtype=np.uint8)
# mask[0:3,0]=1
# mask[2:5,2:4]=1
# roi=cv2.bitwise_and(img,img,mask=mask)
# print('img=\n',img)
# print('mask=\n',mask)
# print('roi=\n',roi)


#分别提取Opencv的logo图像内的红色,绿色,蓝色
#首先利用函数cv2.inRange()查找指定颜色区域,然后利用基于掩码的按位与运算将指定颜色提取出来
# import cv2
# import numpy as np
# opencv=cv2.imread('./image/opencv.jpg')
# hsv=cv2.cvtColor(opencv,cv2.COLOR_BGR2HSV)
# cv2.imshow('opencv',opencv)
# #===============指定蓝色值的范围=================
# minBlue=np.array([110,50,50])
# maxBlue=np.array([130,255,255])
# #确定蓝色区域
# mask=cv2.inRange(hsv,minBlue,maxBlue)
# #通过掩码控制的按位与运算,锁定蓝色区域
# blue=cv2.bitwise_and(opencv,opencv,mask=mask)
# cv2.imshow('blue',blue)
# #================指定绿色值的范围=================
# minGreen=np.array([50,50,50])
# maxGreen=np.array([70,255,255])
# #确定绿色区域
# mask=cv2.inRange(hsv,minGreen,maxGreen)
# #通过掩码控制的按位与运算,锁定绿色区域
# green=cv2.bitwise_and(opencv,opencv,mask=mask)
# cv2.imshow('green',green)
# #==============指定红色值的范围==================
# minRed=np.array([0,50,50])
# maxRed=np.array([30,255,255])
# #确定红色区域
# mask=cv2.inRange(hsv,minRed,maxRed)
# #通过掩码控制的按位与运算,锁定红色区域
# red=cv2.bitwise_and(opencv,opencv,mask=mask)
# cv2.imshow('red',red)
# cv2.waitKey()
# cv2.destroyAllWindows()



#标记肤色,提取一幅图像内的肤色部分
# import cv2
# img=cv2.imread('./image/iu.jpeg')
# hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
# h,s,v=cv2.split(hsv)
# minHue=5
# maxHue=170
# hueMask=cv2.inRange(h,minHue,maxHue)
# minSat=25
# maxSat=166
# satMask=cv2.inRange(s,minSat,maxSat)
# mask=hueMask & satMask
# roi=cv2.bitwise_and(img,img,mask=mask)
# cv2.imshow('img',img)
# cv2.imshow('ROI',roi)
# cv2.waitKey()
# cv2.destroyAllWindows()


#调整HSV色彩空间内V通道的值,观察其处理结果
# import cv2
# img=cv2.imread('./image/iu.jpeg')
# hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
# h,s,v=cv2.split(hsv)
# v[:,:]=255
# newHSV=cv2.merge([h,s,v])
# art=cv2.cvtColor(newHSV,cv2.COLOR_HSV2BGR)
# cv2.imshow('img',img)
# cv2.imshow('art',art)
# cv2.waitKey()
# cv2.destroyAllWindows()


#编写一个程序,分析alpha通道的值
# import cv2
# import numpy as np
# img=np.random.randint(0,256,size=[2,3,3],dtype=np.uint8)
# bgra=cv2.cvtColor(img,cv2.COLOR_BGR2BGRA)
# print('img=\n',img)
# print('bgra=\n',bgra)
# b,g,r,a=cv2.split(bgra)
# print('a=\n',a)
# a[:,:]=125
# bgra=cv2.merge([b,g,r,a])
# print('bgra=\n',bgra)

#编写一个程序,对图像的alpha通道进行处理
# import cv2
# img=cv2.imread('./image/iu.jpeg')
# bgra=cv2.cvtColor(img,cv2.COLOR_BGR2BGRA)
# b,g,r,a=cv2.split(bgra)
# a[:,:]=125
# bgra125=cv2.merge([b,g,r,a])
# a[:,:]=0
# bgra0=cv2.merge([b,g,r,a])
# cv2.imshow('img',img)
# cv2.imshow('bgra',bgra)
# cv2.imshow('bgra125',bgra125)
# cv2.imshow('bgra0',bgra0)
# cv2.waitKey()
# cv2.destroyAllWindows()
# cv2.imwrite('bgra.png',bgra)
# cv2.imwrite('bgra125.png',bgra125)
# cv2.imwrite('bgra0.png',bgra0)


#设计程序,使用函数cv2.resize()来生成一个与原始数组等大小的数组
#根据题目要求
#缩放,在opencv中利用cv2.resize()实现图像的缩放
#情况一:通过参数dsize指定,如果指定了dsize的值,那么无论是否指定参数fx和fy的值,都由dsize
#来标记图像的大小
#dsize内第一个参数对应缩放后图像的宽度,第二个参数对应缩放后图像的高度
# import cv2
# import numpy as np
# img=np.ones([2,4,3],dtype=np.uint8)
# size=img.shape[:2]
# print(size)
# rst=cv2.resize(img,size)
# print('img.shape=\n',img.shape)
# print('img=\n',img)
# print('rst.shape=\n',rst.shape)
# print('rst=\n',rst)


#使用函数cv2.resize()完成一个简单的图像缩放
# import cv2
# img=cv2.imread('./image/iu.jpeg')
# rows,cols=img.shape[:2]#因为是BGR图像,有3个通道,是3维数组
# size=(int(cols*0.9),int(rows*0.5))
# rst=cv2.resize(img,size)
# print('img.shape=',img.shape)
# print('rst.shape=',rst.shape)


#设计程序,控制函数cv2.resize()的fx参数,fy参数.完成图像缩放
#通过参数fx和fy指定,若参数dsize是None,那么通过目标图像的大小通过参数fx和fy来决定
# import cv2
# img=cv2.imread('./image/iu.jpeg')
# rst=cv2.resize(img,None,fx=2,fy=0.5)
# print('img.shape=',img.shape)
# print('rst.shape=',rst.shape)


#设计程序,使用cv2.flip()完成图像的翻转
# import cv2
# img=cv2.imread('./image/iu.jpeg')
# x=cv2.flip(img,0)#绕x轴翻转得到
# y=cv2.flip(img,1)#绕y轴翻转得到
# xy=cv2.flip(img,-1)#围绕x,y轴翻转得到
# cv2.imshow('img',img)
# cv2.imshow('x',x)
# cv2.imshow('y',y)
# cv2.imshow('xy',xy)
# cv2.waitKey()
# cv2.destroyAllWindows()

#仿射变换是指图像经过一系列的几何变换来实现平移,旋转等操作
#仿射图像R=变换矩阵M*原始图像O
# import cv2
# import numpy as np
# img=cv2.imread('./image/iu.jpeg')
# height,width=img.shape[:2]
# x=100
# y=200
# M=np.float32([[1,0,x],[0,1,y]])
# move=cv2.warpAffine(img,M,(width,height))
# cv2.imshow('original',img)
# cv2.imshow('move',move)
# cv2.waitKey()
# cv2.destroyAllWindows()
#仿射变换是指图像经过一系列的几何变换来实现平移,旋转等操作
#仿射图像R=变换矩阵M*原始图像O
# import cv2
# import numpy as np
# img=cv2.imread('./image/iu.jpeg')
# height,width=img.shape[:2]
# x=100
# y=200
# M=np.float32([[1,0,x],[0,1,y]])
# move=cv2.warpAffine(img,M,(width,height))
# cv2.imshow('original',img)
# cv2.imshow('move',move)
# cv2.waitKey()
# cv2.destroyAllWindows()

#设计程序,完成图像旋转
#retval=cv2.getRotationMatrix2D(center,angle,scale)
#center为旋转中心,angle为旋转角度,scale为变换尺度
# import cv2
# img=cv2.imread('./image/iu.jpeg')
# height,width=img.shape[:2]
# #正数逆时针转,负数顺时针转
# M=cv2.getRotationMatrix2D((width/2,height/2),-45,0.6)
# rotate=cv2.warpAffine(img,M,(width,height))
# cv2.imshow('original',img)
# cv2.imshow('rotation',rotate)
# cv2.waitKey()
# cv2.destroyAllWindows()


#设计程序,完成图像仿射
# import cv2
# import numpy as np
# img=cv2.imread('./image/iu.jpeg')
# rows,cols,ch=img.shape
# #src代表输入图像的三个点坐标,dst代表输出图像的三个点坐标
# #记住此处第一个参数对应列数,第二个参数对应行数
# p1=np.float32([[0,0],[cols-1,0],[0,rows-1]])
# p2=np.float32([[0,rows*0.33],[cols*0.85,rows*0.25],[cols*0.15,rows*0.7]])
# M=cv2.getAffineTransform(p1,p2)
# dst=cv2.warpAffine(img,M,(cols,rows))
# cv2.imshow('Original',img)
# cv2.imshow('result',dst)
# cv2.waitKey()
# cv2.destroyAllWindows()


#设计程序,完成图像透视
# import cv2
# import numpy as np
# img=cv2.imread('./image/iu.jpeg')
# rows,cols=img.shape[:2]
# print(rows,cols)
# pts1=np.float32([[150,50],[400,50],[60,450],[310,450]])
# pts2=np.float32([[50,50],[rows-50,50],[50,cols-50],[rows-50,cols-50]])
# #生成函数cv2.warpPerspective()所使用的转换矩阵
# #该函数是cv2.getPerspectiveTransform(),其中第一个参数代表输入图像的四个顶点坐标
# #dst代表输出图像的四个顶点坐标
# M=cv2.getPerspectiveTransform(pts1,pts2)
# #透视变换可以将矩形映射成任意四边形,而前面的仿射变换可以将矩形映射成任意平行四边形
# dst=cv2.warpPerspective(img,M,(cols,rows))
# cv2.imshow('img',img)
# cv2.imshow('dst',dst)
# cv2.waitKey()
# cv2.destroyAllWindows()


#重映射
#把一幅图像内的像素点,放置到另外一幅图像内的指定位置
#dst=cv2.remap(src,map1,map2,interpolation[,borderValue]])
#我们想将目标图像中某个点A映射为原始图像内第x行第y列上的像素点,则将A点所对应的参数map1
#设为y,map2设为x
# import cv2
# import numpy as np
# img=np.random.randint(0,256,(4,5),dtype=np.uint8)
# # print(img)
# rows,cols=img.shape
# mapx=np.ones(img.shape,np.float32)*3
# mapy=np.ones(img.shape,np.float32)*0
# rst=cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
# print('img=\n',img)
# print('mapx=\n',mapx)
# print('mapy=\n',mapy)
# print('rst=\n',rst)



#复制
#通过cv2.remap()函数来实现图像的复制
#设计程序,使用函数cv2.remap()完成数组复制,
# import cv2
# import numpy as np
# img=np.random.randint(0,256,size=[4,5],dtype=np.uint8)
# rows,cols=img.shape
# mapx=np.zeros(img.shape,np.float32)
# mapy=np.zeros(img.shape,np.float32)
# for i in range(rows):
#     for j in range(cols):
#         mapx.itemset((i,j),j)
#         mapy.itemset((i,j),i)
# rst=cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
# print('img=\n',img)
# print('mapx=\n',mapx)
# print('mapy=\n',mapy)
# print('rst=\n',rst)

#设计程序,使用函数cv2.remap()完成图像复制
# import cv2
# import numpy as np
# img=cv2.imread('./image/iu.jpeg')
# rows,cols=img.shape[:2]
# mapx=np.zeros(img.shape[:2],np.float32)
# mapy=np.zeros(img.shape[:2],np.float32)
# for i in range(rows):
#     for j in range(cols):
#         mapx.itemset((i,j),j)
#         mapy.itemset((i,j),i)
# rst=cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
# cv2.imshow('original',img)
# cv2.imshow('result',rst)
# cv2.waitKey()
# cv2.destroyAllWindows()



#设计程序,使用函数cv2.remap()实现数组绕x轴转
#OpenCv中行号的下标是从0开始的,存在对称关系为,'当前行号+对称行号=总行数-1'
#map1的值保持不变,map2的值变为rows-1-i
# import cv2
# import numpy as np
# img=np.random.randint(0,256,size=[4,5],dtype=np.uint8)
# rows,cols=img.shape[:2]
# mapx=np.zeros(img.shape,np.float32)
# mapy=np.zeros(img.shape,np.float32)
# for i in range(rows):
#     for j in range(cols):
#         mapx.itemset((i,j),j)
#         mapy.itemset((i,j),rows-1-i)
# rst=cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
# print('img=\n',img)
# print('mapx=\n',mapx)
# print('mapy=\n',mapy)
# print('rst=\n',rst)

#下面是实际操作
#设计程序,使用函数cv2.remap()实现图像绕x轴的翻转
# import cv2
# import numpy as np
# img=cv2.imread('./image/iu.jpeg')
# rows,cols=img.shape[:2]
# mapx=np.zeros(img.shape[:2],np.float32)
# mapy=np.zeros(img.shape[:2],np.float32)
# for i in range(rows):
#     for j in range(cols):
#         mapx.itemset((i,j),j)
#         mapy.itemset((i,j),rows-1-i)
# rst=cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
# cv2.imshow('original',img)
# cv2.imshow('result',rst)
# cv2.waitKey()
# cv2.destroyAllWindows()


#绕y轴翻转,当前列号+对称列号=总列数-1
# import cv2
# import numpy as np
# img=np.random.randint(0,256,size=[4,5],dtype=np.uint8)
# rows,cols=img.shape
# mapx=np.zeros(img.shape,np.float32)
# mapy=np.zeros(img.shape,np.float32)
# for i in range(rows):
#     for j in range(cols):
#         mapx.itemset((i,j),cols-1-j)
#         mapy.itemset((i,j),i)
# rst=cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
# print('img=\n',img)
# print('mapx=\n',mapx)
# print('mapy=\n',mapy)
# print('rst=\n',rst)


#设计程序,使用函数cv2.remap()实现图像绕y轴的翻转
# import cv2
# import numpy as np
# img=cv2.imread('./image/iu.jpeg')
# rows,cols=img.shape[:2]
# mapx=np.ones(img.shape[:2],np.float32)
# mapy=np.ones(img.shape[:2],np.float32)
# for i in range(rows):
#     for j in range(cols):
#         mapx.itemset((i,j),cols-1-j)
#         mapy.itemset((i,j),i)
# rst=cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
# cv2.imshow('img',img)
# cv2.imshow('rst',rst)
# cv2.waitKey()
# cv2.destroyAllWindows()

#结合上面实现绕x轴和y轴翻转
# import cv2
# import numpy as np
# img=np.random.randint(0,256,size=[4,5],dtype=np.uint8)
# rows,cols=img.shape
# mapx=np.zeros(img.shape,np.float32)
# mapy=np.zeros(img.shape,np.float32)
# for i in range(rows):
#     for j in range(cols):
#         mapx.itemset((i,j),cols-1-j)
#         mapy.itemset((i,j),rows-1-i)
# rst=cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
# print('img=\n',img)
# print('mapx=\n',mapx)
# print('mapy=\n',mapy)
# print('rst=\n',rst)


#设计程序,使用函数cv2.remap()实现图像绕x轴,y轴翻转
# import cv2
# import  numpy as np
# img=cv2.imread('./image/iu.jpeg')
# rows,cols=img.shape[:2]
# mapx=np.zeros(img.shape[:2],np.float32)
# mapy=np.zeros(img.shape[:2],np.float32)
# for i in range(rows):
#     for j in range(cols):
#         mapx.itemset((i,j),cols-1-j)
#         mapy.itemset((i,j),rows-1-i)
# rst=cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
# cv2.imshow('original',img)
# cv2.imshow('rst',rst)
# cv2.waitKey()
# cv2.destroyAllWindows()


#X轴,Y轴互换
#设计程序,使用函数cv2.remap()实现数组的x轴,y轴互换
# import cv2
# import numpy as np
# img=np.random.randint(0,256,size=[4,6],dtype=np.uint8)
# rows,cols=img.shape
# mapx=np.zeros(img.shape,np.float32)
# mapy=np.zeros(img.shape,np.float32)
# for i in range(rows):
#     for j in range(cols):
#         mapx.itemset((i,j),i)
#         mapy.itemset((i,j),j)
# rst=cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
# print('img=\n',img)
# print('mapx=\n',mapx)
# print('mapy=\n',mapy)
# print('rst=\n',rst)
# cv2.waitKey()
# cv2.destroyAllWindows()

#设计程序,使用函数cv2.remap()实现图像的x轴,y轴互换
# import cv2
# import numpy as np
# img=cv2.imread('./image/iu.jpeg')
# rows,cols=img.shape[:2]
# mapx=np.zeros(img.shape[:2],np.float32)
# mapy=np.zeros(img.shape[:2],np.float32)
# for i in range(rows):
#     for j in range(cols):
#         mapx.itemset((i,j),i)
#         mapy.itemset((i,j),j)
# rst=cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
# cv2.imshow('original',img)
# cv2.imshow('result',rst)
# cv2.waitKey()
# cv2.destroyAllWindows()


#图像缩放
#缩小图像后,可以将图像固定在围绕其中心的某个区域
#例如将x轴,y轴设置为:在目标图像的x轴(0.25*x轴长度,0.75*y轴长度)区间内生成缩小图像
#x轴其余区域的点取样自x轴上任意一点的值
#在目标图像的y轴区间内同理
#为了处理方便,让不在上述区域的点都取(0,0)坐标点的值
# import cv2
# import numpy as np
# img=cv2.imread('./image/iu.jpeg')
# rows,cols=img.shape[:2]
# mapx=np.zeros(img.shape[:2],np.float32)
# mapy=np.zeros(img.shape[:2],np.float32)
# for i in range(rows):
#     for j in range(cols):
#         if 0.25*cols< j <0.75*cols and 0.25*rows < i <0.75*rows:
#             mapx.itemset((i,j),2*(j-cols*0.25)+0.5)
#             mapy.itemset((i,j),2*(i-rows*0.25)+0.5)
#         else:
#             mapx.itemset((i,j),0)
#             mapy.itemset((i,j),0)
# rst=cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
# cv2.imshow('original',img)
# cv2.imshow('result',rst)
# cv2.waitKey()
# cv2.destroyAllWindows()


#阈值处理,指剔除图像内像素值高于一定值或者低于一定值的像素点.
#二值化阈值处理(cv2.Thresh_binary)
#所有大于阈值的点处理为最大值,小于或者等于是最小值0
# import cv2
# import numpy as np
# img=np.random.randint(0,256,size=[4,5],dtype=np.uint8)
# t,rst=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
# print('img=\n',img)
# print('t=\n',t)
# print('rst=\n',rst)

#对实际图像的处理
# import cv2
# import numpy as np
# img=cv2.imread('./image/iu.jpeg')
# t,rst=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
# cv2.imshow('img',img)
# cv2.imshow('rst',rst)
# cv2.waitKey()
# cv2.destroyAllWindows()


#反二值化阈值处理
# import cv2
# import numpy as np
# img=np.random.randint(0,256,size=[4,5],dtype=np.uint8)
# t,rst=cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
# print('img=\n',img)
# print('t=\n',t)
# print('rst=\n',rst)


#使用函数cv2.threshold()对图像进行反二值化阈值处理
# import cv2
# import numpy as np
# img=cv2.imread('./image/iu.jpeg')
# t,rst=cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
# cv2.imshow('img',img)
# cv2.imshow('rst',rst)
# cv2.waitKey()
# cv2.destroyAllWindows()



#使用函数cv2.threshhold()对数组进行截断化处理,观察处理结果
# import cv2
# import numpy as np
# img=np.random.randint(0,256,size=[4,5],dtype=np.uint8)
# t,rst=cv2.threshold(img,127,256,cv2.THRESH_TRUNC)
# print('img=\n',img)
# print('t=\n',t)
# print('rst=\n',rst)


#使用函数cv2.threshold()对图像进行截断化阈值处理
# import cv2
# img=cv2.imread('./image/iu.jpeg')
# t,rst=cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
# cv2.imshow('img',img)
# cv2.imshow('rst',rst)
# cv2.waitKey()
# cv2.destroyAllWindows()


#使用函数cv2.threshold()对数组进行超阈值零处理,观察处理结果
# import cv2
# import numpy as np
# img=np.random.randint(0,256,size=[4,5],dtype=np.uint8)
# t,rst=cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)
# print('img=\n',img)
# print('t=\n',t)
# print('rst=\n',rst)


#使用cv2.threshold对图像进行超阈值零处理
# import cv2
# img=cv2.imread('./image/iu.jpeg')
# t,rst=cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)
# cv2.imshow('img',img)
# cv2.imshow('rst',rst)
# cv2.waitKey()
# cv2.destroyAllWindows()


#使用函数cv2.threshhold对数组进行低阈值零处理,观察处理结果
# import cv2
# import numpy as np
# img=np.random.randint(0,256,size=[4,5],dtype=np.uint8)
# t,rst=cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
# print('img=\n',img)
# print('rst=\n',rst)
# cv2.waitKey()
# cv2.destroyAllWindows()

#使用函数cv2.threshold()对图像进行低阈值零处理
# import cv2
# img=cv2.imread('./image/iu.jpeg')
# t,rst=cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
# cv2.imshow('img',img)
# cv2.imshow('rst',rst)
# cv2.waitKey()
# cv2.destroyAllWindows()


#对一幅图像分别使用二值化阈值函数和自适应阈值函数,观察处理结果
# import cv2
# img=cv2.imread('./image/iu.jpeg',0)
# t1,thd=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
# athdMEAN=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,5,3)
# athdGAUS=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,5,3)
# cv2.imshow('img',img)
# cv2.imshow('thd',thd)
# cv2.imshow('athdMEAN',athdMEAN)
# cv2.imshow('athdGAUS',athdGAUS)
# cv2.waitKey()
# cv2.destroyAllWindows()


#测试Otsu阈值处理的实现
# import cv2
# import numpy as np
# img=np.zeros((5,5),dtype=np.uint8)
# img[0:6,0:6]=123
# img[2:6,2:6]=126
# print('img=\n',img)
# t1,thd=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
# print('thd=\n',thd)
# #Otsu方法会遍历所有可能阈值,从而找到最佳阈值
# #使用Otsu方法时,要把阈值设为0,此时函数会寻找最优阈值
# t2,otsu=cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# print(t2)
# print('otsu=\n',otsu)


#分别对一幅图像进行普通的二值化阈值处理和otsu阈值处理,观察处理结果差异
# import cv2
# img=cv2.imread('./image/iu.jpeg',0)
# t1,thd=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
# t2,otsu=cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# cv2.imshow('img',img)
# cv2.imshow('thd',thd)
# cv2.imshow('otsu',otsu)
# cv2.waitKey()
# cv2.destroyAllWindows()


#读取一幅含有噪声的图像,使用cv2.blur()对图像进行均值滤波处理,得到去
#噪图像
# import cv2
# img=cv2.imread('./image/zaosheng.png')
# #该滤波函数有4个参数,src:需要处理的图像,ksize:滤波核的大小,anchor:锚点,bordertype:
# #边界样式
# r=cv2.blur(img,(5,5))
# cv2.imshow('original',img)
# cv2.imshow('res',r)
# cv2.waitKey()
# cv2.destroyAllWindows()

#针对噪声图片,使用不同大小的卷积核对其进行均值滤波,
# import cv2
# img=cv2.imread('./image/zaosheng.png')
# r3=cv2.blur(img,(3,3))
# r8=cv2.blur(img,(8,8))
# cv2.imshow('img',img)
# #使用越大的卷积核,去噪效果越好,失真越严重
# cv2.imshow('r3',r3)
# cv2.imshow('r8',r8)
# cv2.waitKey()
# cv2.destroyAllWindows()

#方框滤波
#有一个参数normalize,表示在进行滤波时是否进行归一化,为1表示要进行
#归一化处理,为0表示不需要,直接使用领域像素值的和即可
# import cv2
# img=cv2.imread('./image/zaosheng.png')
# #默认情况下,normalize的值为1,效果与均值滤波一样
# r=cv2.boxFilter(img,-1,(5,5))
# cv2.imshow('original',img)
# cv2.imshow('rst',r)
# cv2.waitKey()
# cv2.destroyWindow()

# #针对噪声图像,在方框滤波函数cv2.boxFilter()内将参数normalize的值设置为0,显示滤波结果
# import cv2
# img=cv2.imread('./image/zaosheng.png')
# #在本例中没有对图像进行归一化处理
# #进行滤波时计算的是5*5的领域的像素值之和,图像的像素值基本都会超过当前像素值的最大值
# r=cv2.boxFilter(img,-1,(5,5),normalize=0)
# cv2.imshow('original',img)
# cv2.imshow('rst',r)
# cv2.waitKey()
# cv2.destroyWindow()


# import cv2
# img=cv2.imread('./image/zaosheng.png')
# r=cv2.boxFilter(img,-1,(2,2),normalize=0)
# cv2.imshow('original',img)
# cv2.imshow('rst',r)
# cv2.waitKey()
# cv2.destroyWindow()


# import cv2
# img=cv2.imread('./image/zaosheng.png')
# r=cv2.GaussianBlur(img,(5,5),0,0)
# cv2.imshow('original',img)
# cv2.imshow('result',r)
# cv2.waitKey()
# cv2.destroyWindow()

#中值滤波,将领域内的像素值进行排序,然后取中间值,因为要排序,所以耗费时间比较长
# import cv2
# img=cv2.imread('./image/zaosheng.png')
# r=cv2.medianBlur(img,3)
# cv2.imshow('original',img)
# cv2.imshow('rst',r)
# cv2.waitKey()
# cv2.destroyWindow()


#针对噪声图像,对其进行双边滤波
import cv2
img=cv2.imread('./image/zaosheng.png')
#第一个参数src,第二个d表示滤波时选取的空间距离参数,表示以当前像素点为中心的直径
#sigmaColor:滤波处理时选取的颜色差值范围
#sigmaSpace:坐标空间中的sigma值
# r=cv2.bilateralFilter(img,25,100,100)
# cv2.imshow('original',img)
# cv2.imshow('rst',r)
# cv2.waitKey()
# cv2.destroyWindow()

#双边滤波的优势在于对边缘信息的处理
#分别使用高斯滤波以及双边滤波对图片进行处理
# import cv2
# img=cv2.imread('./image/BW.jpg')
# r1=cv2.GaussianBlur(img,(5,5),0,0)
# r2=cv2.bilateralFilter(img,55,100,100)
# cv2.imshow('original',img)
# cv2.imshow('r1',r1)
# cv2.imshow('r2',r2)
# cv2.waitKey()
# cv2.destroyWindow()


#2D卷积
#允许用户使用自定义卷积
# import cv2
# import numpy as np
# cov=np.ones((9,9),np.float32)/81
# img=cv2.imread('./image/zaosheng.png')
# r=cv2.filter2D(img,-1,cov)
# cv2.imshow('original',img)
# cv2.imshow('Guassian',r)
# cv2.waitKey()
# cv2.destroyWindow()



#使用数组演示腐蚀的基本原理
#只有当Kernel的中心点位于img中的img[2,1],img[2,2],img[2,3]
#处时,核才完全处于前景图像中.所以,在腐蚀结果图像中,只有这三个点的值为1,其余点的值皆为0
# import cv2
# import numpy as np
# img=np.zeros((5,5),np.uint8)
# img[1:4,1:4]=1
# kernel=np.ones((3,1),np.uint8)
# erison=cv2.erode(img,kernel)
# print('img=\n',img)
# print('kernel=\n',kernel)
# print('erison=\n',erison)


# #使用cv2.erode()完成图像腐蚀
# #只有完全在前景图像内,才会实现赋值为1
# import cv2
# import numpy as np
# o=cv2.imread('./image/erode.jpg')
# print(o)
# kernel=np.ones((5,5),np.uint8)
# erosion=cv2.erode(o,kernel)
# cv2.imshow('original',o)
# cv2.imshow('erosion',erosion)
# cv2.waitKey()
# cv2.destroyWindow()


#调节函数cv2.erode()的参数,观察不同参数控制下的图像腐蚀效果
# import cv2
# import numpy as np
# o=cv2.imread('./image/erode.jpg',-1)
# kernel=np.ones((9,9),np.uint8)
# #iterations是腐蚀操作迭代的次数,默认为1,表示只进行一次腐蚀操作
# erision=cv2.erode(o,kernel,iterations=5)
# cv2.imshow('original',o)
# cv2.imshow('erosion',erision)
# cv2.waitKey()
# cv2.destroyWindow()

#使用数组演示膨胀的基本原理
#只要当核kernel的任意一点处于前景图像中时,
#就将当前中心点所对应的膨胀结果图像内像素点的值置为1
# import cv2
# import numpy as np
# img=np.zeros((5,5),np.uint8)
# img[2:3,1:4]=1
# kernel=np.ones((3,1),np.uint8)
# dilation=cv2.dilate(img,kernel)
# print('img=\n',img)
# print('kernel=\n',kernel)
# print('dilation=\n',dilation)


#使用函数cv2.dilate()完成图像膨胀操作
# import cv2
# import numpy as np
# img=cv2.imread('./image/erode.jpeg',-1)
# kernel=np.ones((9,9),np.uint8)
# dilation=cv2.dilate(img,kernel)
# cv2.imshow('original',img)
# cv2.imshow('dilation',dilation)
# cv2.waitKey()
# cv2.destroyWindow()


#调节函数cv2.dilate()的参数,实现不同参数控制下图像的膨胀效果
# import cv2
# import numpy as np
# img=cv2.imread('./image/erode.jpeg',-1)
# kernel=np.ones((5,5),np.uint8)
# dilation=cv2.dilate(img,kernel,iterations=9)
# cv2.imshow('original',img)
# cv2.imshow('dilation',dilation)
# cv2.waitKey()
# cv2.destroyWindow()


#通用形态学函数
#将腐蚀和膨胀操作进行组合就可以的得到多种不同形式的运算
#dst=cv2.morphologyEx(src,op,kernel[,iterations[,borderType[,borderValue]]]]])
#开运算:先对图像进行腐蚀,然后对腐蚀的结果进行膨胀,开运算可以用于去噪,计数等
# import cv2
# import numpy as np
# img1=cv2.imread('./image/xintaixue.png')
# img2=cv2.imread('./image/erode.jpeg')
# kernel=np.ones((5,5),np.uint8)
# r1=cv2.morphologyEx(img1,cv2.MORPH_OPEN,kernel)
# r2=cv2.morphologyEx(img2,cv2.MORPH_OPEN,kernel)
# cv2.imshow('img1',img1)
# cv2.imshow('img2',img2)
# cv2.imshow('r1',r1)
# cv2.imshow('r2',r2)
# cv2.waitKey()
# cv2.destroyAllWindows()


#使用函数cv2.morphologyEx()实现闭运算
#就是先膨胀后腐蚀,可用于去除物体上的小黑点以及将不同的前景图像链接起来
# import cv2
# import numpy as np
# img1=cv2.imread('./image/xintaixue.png')
# img2=cv2.imread('./image/erode.jpg')
# k=np.ones((10,10),np.uint8)
# r1=cv2.morphologyEx(img1,cv2.MORPH_CLOSE,k,iterations=3)
# r2=cv2.morphologyEx(img2,cv2.MORPH_CLOSE,k,iterations=3)
# cv2.imshow('img1',img1)
# cv2.imshow('result1',r1)
# cv2.imshow('img2',img2)
# cv2.imshow('result2',r2)
# cv2.waitKey()
# cv2.destroyAllWindows()


#形态学梯度运算:用图像的膨胀图像减去腐蚀图像的操作
#该操作可以获取原始图像中前景图像的边缘.
# import cv2
# import numpy as np
# img=cv2.imread('./image/xintaixue.png',-1)
# k=np.ones((5,5),np.uint8)
# r=cv2.morphologyEx(img,cv2.MORPH_GRADIENT,k)
# cv2.imshow('original',img)
# cv2.imshow('result',r)
# cv2.waitKey()
# cv2.destroyAllWindows()

#使用函数cv2.morphologyEx()实现礼帽运算
#礼帽运算:膨胀图像减去开运算图像,能够获得图像的噪声信息,或者得到比
#原始图像更亮的边缘信息
# import cv2
# import numpy as np
# img1=cv2.imread('./image/xintaixue.png',-1)
# img2=cv2.imread('./image/erode.jpeg',-1)
# k=np.ones((5,5),np.uint8)
# r1=cv2.morphologyEx(img1,cv2.MORPH_TOPHAT,k)
# r2=cv2.morphologyEx(img2,cv2.MORPH_TOPHAT,k)
# cv2.imshow('img1',img1)
# cv2.imshow('r1',r1)
# cv2.imshow('img2',img2)
# cv2.imshow('r2',r2)
# cv2.waitKey()
# cv2.destroyWindow()


#使用函数cv2.morphologyEx()
#黑帽运算:用闭运算减去原始图像的操作,能够获得图像中的小孔或者前景色中的小黑点
#或者得到比原始图像更暗的边缘部分
# import cv2
# import numpy as np
# img1=cv2.imread('./image/xintaixue.png',-1)
# img2=cv2.imread('./image/erode.jpg',-1)
# k=np.ones((5,5),np.uint8)
# r1=cv2.morphologyEx(img1,cv2.MORPH_BLACKHAT,k)
# r2=cv2.morphologyEx(img2,cv2.MORPH_BLACKHAT,k)
# cv2.imshow('original1',img1)
# cv2.imshow('original2',img2)
# cv2.imshow('result1',r1)
# cv2.imshow('result2',r2)
# cv2.waitKey()
# cv2.destroyAllWindows()

#使用函数cv2.getStructuringElement()生成不同结构的核
# import cv2
# k1=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
# k2=cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
# k3=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
# print('kernel1=\n',k1)
# print('kernel2=\n',k2)
# print('kernel3=\n',k3)

#编写程序的,观察不同的核对形态学操作的影响
# import cv2
# import numpy as np
# img=cv2.imread('./image/xintaixue.png',-1)
# k1=cv2.getStructuringElement(cv2.MORPH_RECT,(59,59))
# k2=cv2.getStructuringElement(cv2.MORPH_CROSS,(59,59))
# k3=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(59,59))
# dst1=cv2.dilate(img,k1)
# dst2=cv2.dilate(img,k2)
# dst3=cv2.dilate(img,k3)
# cv2.imshow('original',img)
# cv2.imshow('dst1',dst1)
# cv2.imshow('dst2',dst2)
# cv2.imshow('dst3',dst3)
# cv2.waitKey()
# cv2.destroyAllWindows()

#使用函数cv2.convertScaleAbs()对一个随机数组取绝对值
#opencv中,使用函数cv2.convertScaleAbs()对参数取绝对值,
#dst=cv2.convertScaleAbs(src [,alpha[,beta]])
#alpha代表调节系数,可选;beta是调节亮度值,是默认值,默认为0
# import cv2
# import numpy as np
# img=np.random.randint(-256,256,size=[4,5],dtype=np.int16)
# rst=cv2.convertScaleAbs(img)
# print('img=\n',img)
# print('rst=\n',rst)


#通过实例介绍如何使用函数cv2.Sobel()获取图像边缘信息
# import cv2
# img=cv2.imread('./image/xintaixue.png',0)
# Sobelx=cv2.Sobel(img,-1,1,0)
# cv2.imshow('original',img)
# cv2.imshow('x',Sobelx)
# cv2.waitKey()
# cv2.destroyWindow()


#使用函数cv2.Sobel()获取图像水平方向的完整边缘信息
# import cv2
# img=cv2.imread('./image/xintaixue.png',0)
# Sobelx=cv2.Sobel(img,cv2.CV_64F,1,0)
# Sobelx=cv2.convertScaleAbs(Sobelx)
# cv2.imshow('original',img)
# cv2.imshow('x',Sobelx)
# cv2.waitKey()
# cv2.destroyAllWindows()


#使用函数cv2.Sobel()获取图像垂直方向的边缘信息
#使用函数cv2.convertScaleAbs()对函数cv2.Sobel()的计算结果取绝对值
#获取完整的垂直方向的边缘信息
# import cv2
# img=cv2.imread('./image/xintaixue.png',0)
# Sobely=cv2.Sobel(img,cv2.CV_64F,0,1)
# Sobely=cv2.convertScaleAbs(Sobely)
# cv2.imshow('original',img)
# cv2.imshow('y',Sobely)
# cv2.waitKey()
# cv2.destroyAllWindows()


#设置参数dx和dy的值为"dx=1.dy=1"时,查看执行效果
#这会获取两个方向上的边缘信息,若将ddpeth设置为CV_64F那么就不会截断负值
# import cv2
# img=cv2.imread('./image/xintaixue.png',0)
# Sobelxy=cv2.Sobel(img,cv2.CV_64F,1,1)
# Sobelxy=cv2.convertScaleAbs(Sobelxy)
# cv2.imshow('original',img)
# cv2.imshow('xy',Sobelxy)
# cv2.waitKey()
# cv2.destroyAllWindows()

#计算函数cv2.Sobel()在水平,垂直两个方向叠加的边缘信息
# import cv2
# img=cv2.imread('./image/xintaixue.png',-1)
# Sobelx=cv2.Sobel(img,cv2.CV_64F,1,0)
# Sobely=cv2.Sobel(img,cv2.CV_64F,0,1)
# Sobelx=cv2.convertScaleAbs(Sobelx)
# Sobely=cv2.convertScaleAbs(Sobely)
# Sobelxy=cv2.addWeighted(Sobelx,0.5,Sobely,0.5,0)
# cv2.imshow('original',img)
# cv2.imshow('xy',Sobelxy)
# cv2.waitKey()
# cv2.destroyAllWindows()


#使用2种方法处理图像在两个方向的边缘信息
# import cv2
# img=cv2.imread('./image/iu.jpeg',0)
# Sobelx=cv2.Sobel(img,cv2.CV_64F,1,0)
# Sobely=cv2.Sobel(img,cv2.CV_64F,0,1)
# Sobelx=cv2.convertScaleAbs(Sobelx)
# Sobely=cv2.convertScaleAbs(Sobely)
# Sobelxy=cv2.addWeighted(Sobelx,0.5,Sobely,0.5,0)
# Sobelxy11=cv2.Sobel(img,cv2.CV_64F,1,1)
# Sobelxy11=cv2.convertScaleAbs(Sobelxy11)
# cv2.imshow('original',img)
# cv2.imshow('xy',Sobelxy)
# cv2.imshow('xy11',Sobelxy11)
# cv2.waitKey()
# cv2.destroyAllWindows()


#使用函数cv2.Scharr()获取图像水平方向的边缘信息
# import cv2
# img=cv2.imread('./image/xintaixue.png',0)
# Scharrx=cv2.Scharr(img,cv2.CV_64F,1,0)
# Scharrx=cv2.convertScaleAbs(Scharrx)
# cv2.imshow('original',img)
# cv2.imshow('Scharrx',Scharrx)
# cv2.waitKey()
# cv2.destroyAllWindows()


#使用函数cv2.Scharr()获取图像垂直方向的边缘信息
# import cv2
# img=cv2.imread('./image/xintaixue.png',0)
# Scharry=cv2.Scharr(img,cv2.CV_64F,0,1)
# Scharry=cv2.convertScaleAbs(Scharry)
# cv2.imshow('original',img)
# cv2.imshow('Scharrx',Scharry)
# cv2.waitKey()
# cv2.destroyAllWindows()


#使用函数cv2.Scharr()实现水平方向和垂直方向边缘叠加的效果
# import cv2
# img=cv2.imread('./image/iu.jpeg',0)
# scharrx=cv2.Scharr(img,cv2.CV_64F,1,0)
# scharry=cv2.Scharr(img,cv2.CV_64F,0,1)
# scharrx=cv2.convertScaleAbs(scharrx)
# scharry=cv2.convertScaleAbs(scharry)
# scharrxy=cv2.addWeighted(scharrx,0.5,scharry,0.5,0)
# cv2.imshow('original',img)
# cv2.imshow('xy',scharrxy)
# cv2.waitKey()
# cv2.destroyAllWindows()


#观察将函数cv2.Scharr()的参数dx,dy同时设置为1时,程序的运行情况
# import cv2
# import numpy as np
# img=cv2.imread('./image/iu.jpeg',0)
# #不允许同时将dx与dy设置为1
# Scharrxy11=cv2.Scharr(img,cv2.CV_64F,1,1)
# Scharrxy11=cv2.convertScaleAbs(Scharrxy11)
# cv2.imshow('original',img)
# cv2.imshow('Scharrxy11',Scharrxy11)
# cv2.waitKey()
# cv2.destroyAllWindows()

#使用函数cv2.Sobel()中ksize的参数值为-1时,就会使用Scharr算子进行运算
# import cv2
# img=cv2.imread('./image/xintaixue.png',0)
# Scharrx=cv2.Sobel(img,cv2.CV_64F,1,0,-1)
# Scharry=cv2.Sobel(img,cv2.CV_64F,0,1,-1)
# Scharrx=cv2.convertScaleAbs(Scharrx)
# Scharry=cv2.convertScaleAbs(Scharry)
# cv2.imshow('original',img)
# cv2.imshow('Scharrx',Scharrx)
# cv2.imshow('Scharry',Scharry)
# cv2.waitKey()
# cv2.destroyAllWindows()


#Sobel算子和Scharr算子的比较
#Sobel算子的缺点是,当其核结构较小时,精确度不高,而Scharr算子有更高的精确度
#分别使用Sobel算子和Scharr算子来计算一幅图像的水平边缘和垂直边缘的叠加信息
# import cv2
# img=cv2.imread('./image/iu.jpeg',0)
# Sobelx=cv2.Sobel(img,cv2.CV_64F,1,0)
# Sobelx=cv2.convertScaleAbs(Sobelx)
# Sobely=cv2.Sobel(img,cv2.CV_64F,0,1)
# Sobely=cv2.convertScaleAbs(Sobely)
# Scharrx=cv2.Scharr(img,cv2.CV_64F,1,0)
# Scharrx=cv2.convertScaleAbs(Scharrx)
# Scharry=cv2.Scharr(img,cv2.CV_64F,0,1)
# Scharry=cv2.convertScaleAbs(Scharry)
# cv2.imshow('original',img)
# cv2.imshow('Sobelx',Sobelx)
# cv2.imshow('Sobely',Sobely)
# cv2.imshow('Scharrx',Scharrx)
# cv2.imshow('Scharry',Scharry)
# cv2.waitKey()
# cv2.destroyAllWindows()


#使用函数cv2.Laplacian()计算图像的边缘信息
# import cv2
# img=cv2.imread('./image/iu.jpeg',0)
# Laplacian=cv2.Laplacian(img,cv2.CV_64F)
# Laplacian=cv2.convertScaleAbs(Laplacian)
# cv2.imshow('original',img)
# cv2.imshow('Laplacian',Laplacian)
# cv2.waitKey()
# cv2.destroyAllWindows()


#Canny函数及使用
# import cv2
# img=cv2.imread('./image/iu.jpeg',-1)
# r1=cv2.Canny(img,128,200)
# r2=cv2.Canny(img,32,128)
# cv2.imshow('original',img)
# cv2.imshow('r1',r1)
# cv2.imshow('r2',r2)
# cv2.waitKey()
# cv2.destroyAllWindows()


#使用函数cv2.pyrDown()对一幅图像进行向下采样,观察采样结果
# import cv2
# img=cv2.imread('./image/iu.jpeg',0)
# r1=cv2.pyrDown(img)
# r2=cv2.pyrDown(r1)
# r3=cv2.pyrDown(r2)
# print('img.shape=\n',img.shape)
# print('r1.shape=\n',r1.shape)
# print('r2.shape=\n',r2.shape)
# print('r3.shape=\n',r3.shape)
# cv2.imshow('original',img)
# cv2.imshow('r1',r1)
# cv2.imshow('r2',r2)
# cv2.imshow('r3',r3)
# cv2.waitKey()
# cv2.destroyAllWindows()


#使用函数cv2.pyrUp()对衣服图像进行向上采样,观察采样的结果
#对图像进行向上采样的时候,在每个像素的右侧,下方分别插入零值列和零值行
#然后使用高斯滤波对新图像进行滤波,得到向上采样的结果图像
# import cv2
# img=cv2.imread('./image/zaosheng.png',0)
# r1=cv2.pyrUp(img)
# r2=cv2.pyrUp(r1)
# r3=cv2.pyrUp(r2)
# print('img.shape=\n',img.shape)
# print('r1.shape=\n',r1.shape)
# print('r2.shape=\n',r2.shape)
# print('r3.shape=\n',r3.shape)
# cv2.imshow('original',img)
# cv2.imshow('r1',r1)
# cv2.imshow('r2',r2)
# cv2.imshow('r3',r3)
# cv2.waitKey()
# cv2.destroyAllWindows()

#使用函数cv2.pyrDown()和函数cv2.pyrUp(),先后对一幅图像进行向下采样,向上采样
#,观察采样的结果及结果图像与原始图像的差异
# import cv2
# img=cv2.imread('./image/iu.jpeg')
# down=cv2.pyrDown(img)
# up=cv2.pyrUp(down)
# #原始图像与先后经过向下采样,向上采样得到的结果图像,在大小上是相等的
# #在外观上是相似的,但是他们的像素值并不一致
# diff=up-img
# print('img.shape=\n',img.shape)
# print('up.shape=\n',up.shape)
# cv2.imshow('original',img)
# cv2.imshow('up',up)
# cv2.imshow('diff',diff)
# cv2.waitKey()
# cv2.destroyAllWindows()



#与上面正好相反,先经过向上采样,再经过向下采样
# import cv2
# img=cv2.imread('./image/iu.jpeg')
# up=cv2.pyrUp(img)
# down=cv2.pyrDown(up)
# #原始图像与先后经过向上采样,向下采样得到的结果图像,在大小上是相等的
# #在外观上是相似的,但是他们的像素值并不一致
# diff=down-img
# print('img.shape=\n',img.shape)
# print('down.shape=\n',down.shape)
# cv2.imshow('original',img)
# cv2.imshow('down',down)
# cv2.imshow('diff',diff)
# cv2.waitKey()
# cv2.destroyAllWindows()


#使用函数cv2.pyrDown()以及函数cv2.pyrUp()构造拉普拉斯金字塔
# import cv2
# img=cv2.imread('./image/test2.jpeg')
# G0=img
# G1=cv2.pyrDown(G0)
# G2=cv2.pyrDown(G1)
# G3=cv2.pyrDown(G2)
# L0=G0-cv2.pyrUp(G1)
# L1=G1-cv2.pyrUp(G2)
# L2=G2-cv2.pyrUp(G3)
# print('L0.shape=\n',L0.shape)
# print('L1.shape=\n',L1.shape)
# print('L2.shape=\n',L2.shape)
# cv2.imshow('original',img)
# cv2.imshow('L0',L0)
# cv2.imshow('L1',L1)
# cv2.imshow('L2',L2)
# cv2.waitKey()
# cv2.destroyAllWindows()


#编写程序,使用拉普拉斯金字塔及高斯金字塔恢复原始图像
# import cv2
# import numpy as np
# img=cv2.imread('./image/iu.jpeg')
# G0=img
# G1=cv2.pyrDown(G0)
# L0=G0-cv2.pyrUp(G1)
# R0=L0+cv2.pyrUp(G1)#通过拉普拉斯图像回复原始图像
# print('img.shape=\n',img.shape)
# print('R0.shape=\n',R0.shape)
# result=R0-img#将img和R0做减法运算
# #计算result的绝对值,避免求和时负负为正
# result=abs(result)
# #计算result所有元素的和
# print('原始图像与恢复图像之间的差',np.sum(result))

#编写程序,使用拉普拉斯金字塔及高斯金字塔恢复高斯金字塔内的多层图像
#每一层原始图像与恢复图像都一样
# import cv2
# import numpy as np
# img=cv2.imread('./image/BW.jpg')
# #==============生成高斯金字塔==================
# G0=img
# G1=cv2.pyrDown(G0)
# G2=cv2.pyrDown(G1)
# G3=cv2.pyrDown(G2)
# #===========生成拉普拉斯金字塔==============
# L0=G0-cv2.pyrUp(G1)#拉普拉斯金字塔第0层
# L1=G1-cv2.pyrUp(G2)
# L2=G2-cv2.pyrUp(G3)
# #=================复原G0====================
# RG0=L0+cv2.pyrUp(G1)#通过拉普拉斯金字塔恢复原图像G0
# print('G0.shape=\n',G0.shape)
# print('RG0.shape=\n',RG0.shape)
# result=RG0-G0
# #计算result的绝对值时,避免求和时负负为正,1+(-1)=0
# result=abs(result)
# #计算result所有元素的和
# print('原始图像G0与恢复图像RG0差值的绝对值和:',np.sum(result))
# #=================复原G1====================
# RG1=L1+cv2.pyrUp(G2)#通过拉普拉斯金字塔恢复原图像G0
# print('G1.shape=\n',G1.shape)
# print('RG1.shape=\n',RG1.shape)
# result=RG1-G1
# #计算result的绝对值时,避免求和时负负为正,1+(-1)=0
# result=abs(result)
# #计算result所有元素的和
# print('原始图像G1与恢复图像RG1差值的绝对值和:',np.sum(result))
# #=================复原G2====================
# RG2=L2+cv2.pyrUp(G3)#通过拉普拉斯金字塔恢复原图像G0
# print('G2.shape=\n',G2.shape)
# print('RG2.shape=\n',RG2.shape)
# result=RG2-G2
# #计算result的绝对值时,避免求和时负负为正,1+(-1)=0
# result=abs(result)
# #计算result所有元素的和
# print('原始图像G2与恢复图像RG2差值的绝对值和:',np.sum(result))

#图像轮廓
#绘制一幅图像内的所有轮廓
# import cv2
# img=cv2.imread('./image/1.jpg')
# cv2.imshow('original',img)
# gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# ret,binary=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
# contours, hierarchy =cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
# #参数依次表示待绘制轮廓的图像,需要绘制的轮廓,需要绘制的边缘索引,绘制的颜色,画笔的粗细
# img=cv2.drawContours(img,contours,-1,(0,0,255),1)
# cv2.imshow('result',img)
# cv2.waitKey()
# cv2.destroyAllWindows()

#逐个显示一幅图像内的边缘信息
#需要设置contoursIdx的值
# import cv2
# import numpy as np
# img=cv2.imread('./image/1.jpg')
# cv2.imshow('original',img)
# gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# ret,binary=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
# contours,hierarchy=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
# n=len(contours)
# contoursImg=[]
# for i in range(n):
#     temp=np.zeros(img.shape,np.uint8)
#     contoursImg.append(temp)
#     contoursImg[i]=cv2.drawContours(
#         contoursImg[i],contours,i,(255,255,255),5
#     )
#     cv2.imshow('contours['+str(i)+']',contoursImg[i])
# cv2.waitKey()
# cv2.destroyAllWindows()

#使用轮廓绘制功能,提取前景图像
#将函数cv2.drawContours()的参数thickness设置为-1,绘制实心轮廓
#将实心轮廓与原始图像进行按位与原始图像,即可将前景对象从原始图像中拉取出来
# import cv2
# import numpy as np
# img=cv2.imread('./image/feather.jpg')
# cv2.imshow('original',img)
# gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# ret,binary=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
# contours,hierarchy=cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
# mask=np.zeros(img.shape,np.uint8)
# #在黑色背景图片上绘制外轮廓
# mask=cv2.drawContours(mask,contours,-1,(255,255,255),-1)
# cv2.imshow('mask',mask)
# loc=cv2.bitwise_and(img,mask)
# cv2.imshow('location',loc)
# cv2.waitKey()
# cv2.destroyAllWindows()

#矩的计算:moments函数
#在opencv中,函数cv2.moments()同时会计算上述空间矩
#中心矩,归一化中心距
#使用函数cv2.moments()提取一幅图像的特征
# import cv2
# import numpy as np
# img=cv2.imread('./image/feather.jpg')
# cv2.imshow('original',img)
# # print(img.shape)
# gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# # print(gray)
# #阈值处理
# ret,binary=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
# contours,hierarchy=cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
# n=len(contours)
# contoursImg=[]
# # print(n)
# for i in range(n):
#     temp=np.zeros(binary.shape,np.uint8)
#     contoursImg.append(temp)
#     contoursImg[i]=cv2.drawContours(contoursImg[i],contours,i,255,3)
#     cv2.imshow('contours['+str(i)+']',contoursImg[i])
# print('观察各个轮廓的矩:(moments):')
# for i in range(n):
#     print('轮廓'+str(i)+'的矩:\n',cv2.moments(contours[i]))
# print('观察各个轮廓的面积:')
# for i in range(n):
#     print('轮廓'+str(i)+'的面积:%d'%cv2.moments(contours[i])['m00'])
# cv2.waitKey()
# cv2.destroyAllWindows()

#计算轮廓的面积:contourArea()函数
# import cv2
# import numpy as np
# img=cv2.imread('./image/feather.jpg')
# gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# ret,binary=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
# contours,hierarchy=cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
# cv2.imshow('original',img)
# n=len(contours)
# contoursImg=[]
# for i in range(n):
#     #函数cv2.contourArea()有第二个参数oriented,当它为True时,返回的值包含正/负号,
#     #用来表示轮廓是顺时针还是逆时针默认为false,返回绝对值
#     #将面积大于15000的轮廓筛选出来
#     if(cv2.contourArea(contours[i])>15000):
#         print('Contours['+str(i)+']面积=',cv2.contourArea(contours[i]))
#     temp=np.zeros(img.shape,np.uint8)
#     contoursImg.append(temp)
#     contoursImg[i]=cv2.drawContours(contoursImg[i],
#                                     contours,
#                                     i,
#                                     (255,255,255),
#                                     3)
#     cv2.imshow('contours['+str(i)+']',contoursImg[i])
#
# cv2.waitKey()
# cv2.destroyAllWindows()


#计算轮廓的长度:arcLength函数,美丽的蝴蝶图片
# import cv2
# import numpy as np
# #-----------------读取及显示原始图像------------------
# img=cv2.imread('./image/2.jpg')
# cv2.imshow('original',img)
# #=---------------获取轮廓-------------------------------
# gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# ret,binary=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
# contours,hierarchy=cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
# #----------------计算各轮廓的长度之和,平均长度-----------------
# n=len(contours)#获取轮廓的个数
# cntLen=[]#存储各轮廓的长度
# for i in range(n):
#     cntLen.append(cv2.arcLength(contours[i],True))
#     print('第'+str(i)+'个轮廓的长度:%d'%cntLen[i])
# cntLenSum=np.sum(cntLen)#各轮廓的长度之和
# cntLenAvr=cntLenSum/n #轮廓长度的平均值
# print('轮廓的总长度为:%d'%cntLenSum)
# print('轮廓的平均长度:%d'%cntLenAvr)
# #---------------显示长度超过平均值的轮廓-------------------
# contoursImg=[]
# for i in range(n):
#     temp=np.zeros(img.shape,np.uint8)
#     contoursImg.append(temp)
#     contoursImg[i]=cv2.drawContours(contoursImg[i],contours,i,(255,255,255),3)
#     if cv2.arcLength(contours[i],True)>cntLenAvr:
#         cv2.imshow('Contours['+str(i)+']',contoursImg[i])
# cv2.waitKey()
# cv2.destroyAllWindows()



#计算图像的Hu矩,对其中第0个矩的关系进行演示
#h0=v20+v02
# i

# import cv2
# #-----------------计算图像o1的Hu矩阵----------------
# img1=cv2.imread('./image/iu.jpeg')
# gray1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
# HuM1=cv2.HuMoments(cv2.moments(gray1)).flatten()
# #-----------------计算图像o2的Hu矩阵----------------
# img2=cv2.imread('./image/ym.jpeg')
# gray2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
# HuM2=cv2.HuMoments(cv2.moments(gray2)).flatten()
# #-----------------计算图像o3的Hu矩阵----------------
# img3=cv2.imread('./image/erode.jpeg')
# gray3=cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY)
# HuM3=cv2.HuMoments(cv2.moments(gray3)).flatten()
# #----------------打印图像img1,img2,img3------------
# print('img1.shape=',img1.shape)
# print('img2.shape=',img2.shape)
# print('img3.shape=',img3.shape)
# print('cv2.moments(gray1)=\n',cv2.moments(gray1))
# print('cv2.moments(gray1)=\n',cv2.moments(gray2))
# print('cv2.moments(gray2)=\n',cv2.moments(gray3))
# print('\nHuM1=\n',HuM1)
# print('\nHuM2=\n',HuM2)
# print('\nHuM3=\n',HuM3)
# #-----------------计算图像img1与图像img2,图像img3的Hu矩之差-------------------
# print('\nHuM1-HuM2=',HuM1-HuM2)
# print('\nHuM1-HuM3=',HuM1-HuM3)
# #--------------显示图像---------------
# cv2.imshow('original1',img1)
# cv2.imshow('original2',img2)
# cv2.imshow('original3',img3)
# cv2.waitKey()
# cv2.destroyAllWindows()


#使用函数cv2.matchShapes()计算三幅不同图像的匹配值
# import cv2
# img1=cv2.imread('./image/iu.jpeg')
# img2=cv2.imread('./image/ym.jpeg')
# gray1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
# gray2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
# ret,binary1=cv2.threshold(gray1,127,255,cv2.THRESH_BINARY)
# ret,binary2=cv2.threshold(gray2,127,255,cv2.THRESH_BINARY)
# contours1,hierarchy=cv2.findContours(binary1,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
# contours2,hierarchy=cv2.findContours(binary2,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
# cnt1=contours1[0]
# cnt2=contours2[0]
# ret0=cv2.matchShapes(cnt1,cnt1,1,0.0)
# ret1=cv2.matchShapes(cnt1,cnt2,1,0.0)
# print('相同图像的matchShape=',ret0)
# print('不同图像的matchShape=',ret1)


#轮廓拟合
#在计算轮廓的时候,有时并不需要实际的轮廓,而是仅仅需要一个接近于轮廓的近似多边形
#设计程序,显示函数cv2.boundingRect()不同形式的返回值
# import cv2
# #-----------读取并显示原始图像----------------
# img=cv2.imread('./image/2.jpg')
# #-----------提取图像轮廓-------------------
# gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# ret,binary=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
# contours,hierarchy=cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
# #----------返回顶点及边长------------------
# #4个返回值分别是:
# #矩阵边界左上角顶点x坐标,矩阵左上角顶点的y坐标
# #矩形边界的x方向的长度,y方向的长度
# x,y,w,h=cv2.boundingRect(contours[0])
# print('顶点及长宽的点形式:')
# print('x=',x)
# print('y=',y)
# print('w=',w)
# print('h=',h)
# #------------仅有一个返回值的情况---------------
# rect=cv2.boundingRect(contours[0])
# print('\n顶点及长宽的元组(tuple)形式\n')
# print('rect=',rect)


#使用函数cv2.drawContours()绘制矩形包围框
# import cv2
# import numpy as np
# #-------------获取并显示原始图像---------------------
# img=cv2.imread('./image/2.jpg')
# cv2.imshow('original',img)
# #-------------提取图像轮廓-------------------------
# gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#使用函数cv2.drawContours()绘制矩形包围框
# ret,binary=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
# contours,hierarchy=cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
# #--------------构造矩形边界-----------------------
# x,y,w,h=cv2.boundingRect(contours[0])
# brcnt=np.array([[[x,y]],[[x+w,y]],[[x+w,y+h]],[[x,y+h]]])
# cv2.drawContours(img,[brcnt],-1,(255,255,255),2)
# #--------------显示矩形边界---------------------
# cv2.imshow('result',img)
# cv2.waitKey()
# cv2.destroyAllWindows()


#使用函数cv2.boundingRect()及cv2.rectangle()绘制矩形包围框
# import cv2
# #----------------读取并显示原始图像-------------
# img=cv2.imread('./image/2.jpg')
# cv2.imshow('original',img)
# #----------------提取图像轮廓-------------------
# gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# ret,binary=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
# #查找图像轮廓函数的参数:image,mode,method,依次为原始图像,lun
# contours,hierarchy=cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
# #-------------构造矩形边界----------------------
# x,y,w,h=cv2.boundingRect(contours[0])
# cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
# #-------------显示矩形边界--------------------
# cv2.imshow('result',img)
# cv2.waitKey()
# cv2.destroyAllWindows()


#使用函数cv2.minAreaRect()就算图像的最小包围矩形框
# import cv2
# import numpy as np
# img=cv2.imread('./image/2.jpg')
# cv2.imshow('original',img)
# gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# ret,binary=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
# contours,hierarchy=cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
# rect=cv2.minAreaRect(contours[0])
# print('返回值rect:\n',rect)
# points=cv2.boxPoints(rect)
# print('\n转换后的points:\n',points)
# points=np.int0(points)#取整
# image=cv2.drawContours(img,[points],0,(255,255,255),2)
# cv2.imshow('result',img)
# cv2.waitKey()
# cv2.destroyAllWindows()


#最小包围圆形,使用函数cv2.minEnclosingCircle()构造图像的最小包围圆形
# import cv2
# img=cv2.imread('./image/2.jpg')
# cv2.imshow('orginal',img)
# gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# ret,binary=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
# contours,hierarchy=cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
# #函数cv2.minEnclosingCircle()通过迭代算法构造一个对象的面积最小包围圆形
# #语法格式为:center,radius=cv2.minEnclosingCircle(points)
# (x,y),radius=cv2.minEnclosingCircle(contours[3])
# center=(int(x),int(y))
# radius=int(radius)
# cv2.circle(img,center,radius,(255,255,255),2)
# cv2.imshow('result',img)
# cv2.waitKey()
# cv2.destroyAllWindows()

#构造最优拟合椭圆.
# import cv2
# img=cv2.imread('./image/2.jpg')
# gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# ret,binary=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
# contours,hierarchy=cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
# cv2.imshow('original',img)
# ellipse=cv2.fitEllipse(contours[4])
# print('ellipse=',ellipse)
# cv2.ellipse(img,ellipse,(0,255,0),3)
# cv2.imshow('result',img)
# cv2.waitKey()
# cv2.destroyAllWindows()


#翻车
#使用函数cv2.fitLine()构造最优拟合直线
# import cv2
# img=cv2.imread('./image/2.jpg')
# cv2.imshow('original',img)
# gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# ret,binary=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
# contours,hierarchy=cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
# rows,cols=img[:2]
# [vx,vy,x,y]=cv2.fitLine(contours[0],cv2.DIST_L2,0,0.01,0.01)
# lefty=int((-x*vy/vx)+y)
# righty=int(((cols-x)*vy/vx)+y)
# cv2.line(img,(cols-1,righty),(0,lefty),(0,255,0),2)
# cv2.imshow('result',img)
# cv2.waitKey()
# cv2.destroyAllWindows()

#使用函数cv2.minEnclosingTriangle()构造最小外包三角形
# import cv2
# img=cv2.imread('./image/2.jpg')
# cv2.imshow('original',img)
# gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# ret,binary=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
# contours,hierarchy=cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
# area,trg1=cv2.minEnclosingTriangle(contours[0])
# print('area=',area)
# print('trg1=',trg1)
# for i in range(0,3):
#     cv2.line(img,tuple(trg1[i][0]),tuple(trg1[(i+1)%3][0]),(255,255,255),2)
# cv2.imshow('result',img)
# cv2.waitKey()
# cv2.destroyAllWindows()

#使用函数cv2.approxPolyDp()构造不同精度的逼近多边形
import cv2
#---------------------读取并显示原始图像-----------------------
img=cv2.imread('./image/2.jpg')
cv2.imshow('original',img)
#---------------------获取轮廓------------------------------
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,binary=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
contours,hierarchy=cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
#-------------------epsilon=0.1*周长----------------------
adp=img.copy()
epsilon=0.1*cv2.arcLength(contours[3],True)
approx=cv2.approxPolyDP(contours[3],epsilon,True)
adp=cv2.drawContours(adp,[approx],0,(0,0,255),2)
cv2.imshow('result0.1',adp)
#-------------------epsilon=0.09*周长----------------------
adp=img.copy()
epsilon=0.09*cv2.arcLength(contours[3],True)
approx=cv2.approxPolyDP(contours[3],epsilon,True)
adp=cv2.drawContours(adp,[approx],0,(0,0,255),2)
cv2.imshow('result0.09',adp)
#-------------------epsilon=0.1*周长----------------------
adp=img.copy()
epsilon=0.1*cv2.arcLength(contours[3],True)
approx=cv2.approxPolyDP(contours[3],epsilon,True)
adp=cv2.drawContours(adp,[approx],0,(0,0,255),2)
cv2.imshow('result01.',adp)
#-------------------epsilon=0.055*周长----------------------
adp=img.copy()
epsilon=0.055*cv2.arcLength(contours[3],True)
approx=cv2.approxPolyDP(contours[3],epsilon,True)
adp=cv2.drawContours(adp,[approx],0,(0,0,255),2)
cv2.imshow('result0.055',adp)
#-------------------epsilon=0.05*周长----------------------
adp=img.copy()
epsilon=0.05*cv2.arcLength(contours[3],True)
approx=cv2.approxPolyDP(contours[3],epsilon,True)
adp=cv2.drawContours(adp,[approx],0,(0,0,255),2)
cv2.imshow('result0.05',adp)
#-------------------epsilon=0.02*周长----------------------
adp=img.copy()
epsilon=0.02*cv2.arcLength(contours[3],True)
approx=cv2.approxPolyDP(contours[3],epsilon,True)
adp=cv2.drawContours(adp,[approx],0,(0,0,255),2)
cv2.imshow('result0.02',adp)
#----------------------------等待释放窗口------------------
cv2.waitKey()
cv2.destroyAllWindows()