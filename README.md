# 2023SummerCamp
浙江大学混合现实中心夏令营  
选题:室外场景重建以及Relighting  
任务：在NeuconW基础上，在重建与渲染过程中加入光照编码，实现几何和光照的同时恢复。  

所有代码在zju3dv NeuralRecon-W基础上添加，见commit history  
*添加的内容（更新中...）：
1.根据Nerf-W的结构，引入Appearance Embedding来编码光照变化，创建了一个LightCodeNetwork
2.训练LightNeuconW模型，得到ckpt后尝试对测试集数据进行rendering，并通过appearance embedding表示光照，控制图像渲染的过程
3.加入了配套的Model，System，config，Train等代码