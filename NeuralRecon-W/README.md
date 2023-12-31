# 2023SummerCamp
浙江大学混合现实中心夏令营  
选题:室外场景重建以及Relighting  
任务：在NeuconW基础上，在重建与渲染过程中加入光照编码，实现几何和光照的同时恢复。  

所有代码在zju3dv NeuralRecon-W基础上添加，见commit history  

## git分支说明:  
**master(defualt):**开发的主分支，完成了基本任务--重建几何表面并通过光照code控制图像渲染  
**osr:**进阶任务，根据Nerf-OSR实现新的渲染方式，恢复几何表面和材质

## 添加的内容:  
1.根据Nerf-W的结构，引入Appearance Embedding来编码光照变化，创建了一个*LightCodeNetwork*；根据DeepSDF的结构，创建了类似的*SDFNetwork*; 将以上两者结合为*LightNeuconW*  
2.编写了配套的Model，PytorchLightning System，config，Train等代码  
3.训练LightNeuconW模型，得到ckpt后尝试对测试集数据进行rendering，并通过appearance embedding表示光照，控制图像渲染的过程，对相同场景使用不同光线进行relighting  
4.根据*NeRF-OSR*，创建新的渲染方式、并将shadow和albedo加入LightCodeNetwork，尝试恢复场景的反射、阴影等材质信息  
5.尝试使用新的渲染方式，用阴影、反射率等控制relighting（进行中）  

## 文件结构  
|--model  
|------light_neuconw.py(**NEW**,定义了新的神经网络结构,引入光照)  
|--rendering  
|------custom_rendering.py(**NEW**,定义了新的渲染方式,光照和OSR)  
|--config  
|------defaults.py(**MODIFY**,修改配置文件)  
|--scripts  
|------custom_train.sh(**NEW**,单机单卡训练脚本)  
|--custom_train.py(**NEW**,训练代码：定义了Pytorch Lightning System)  
|--test_light.py(**NEW**,测试代码：在ckpt上测试用光照控制渲染)  
|--visualization.py(**NEW**,可视化点云)  
|--losses.py(**MODIFY**,为新模型定义新的损失函数)  
