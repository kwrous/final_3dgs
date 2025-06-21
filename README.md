# final_3dgs
3D Gaussian Splatting实现\
数据集同 [Tkwrous/final_nerf: 原版nerf实现](https://github.com/kwrous/final_nerf) \
克隆仓库：https://github.com/graphdeco-inria/gaussian-splatting \
将data以及change_route.py 放入\
安装相关依赖\
在终端中运行python train.py -s data -m data\output --eval  进行训练\
运行change_route.py生成新的轨迹下的images.bin\
在终端中运行python render.py -m data\output 进行新轨迹上的渲染\
使用ffmpeg -framerate 24 -start_number 0 -i /root/gaussian-splatting/data/output/train/ours_30000/renders/%05d.png -c:v libx264 -pix_fmt yuv420p /root/gaussian-splatting/output_video.mp4生成渲染视频output.video\
运行python metrics.py -m data\output 进行评价指标计算。
