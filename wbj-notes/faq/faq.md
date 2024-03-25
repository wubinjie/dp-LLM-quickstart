
# FAQ 


## 1. 华为云环境搭建问题

注意升级到Ubuntu 22.04 版本， 使用 cuda toolkit 最新版本（12.4）直接安装驱动。

## 2. 教程代码链接
https://github.com/DjangoPeng/LLM-quickstart

## 3. 显存计算问题

在后续的课程中有介绍。


## 4. ipynb文件如何转成py文件 
jupyter nbconvert --to script fine-tune-quickstart-all.ipynb

## 5. 磁盘空间不足问题
推荐至少使用300GB的云硬盘。

## 6. hugging face 如何下载模型 
参考这个文档 https://zhuanlan.zhihu.com/p/678611989


## 7. 如何将 jupyter 放到后台执行
nohup jupyter lab --allow-root > jupyterlab.log 2>&1 &

## 8. 如何关闭后台执行的jupyter

找出包含jupyter的进程

```shell 
ps -aux | grep jupyter
```

会列出很多，然后找到 jupyter lab相关的进程， 一般情况下第二列就是pid, 如，我的进程是 9608，然后再用如下命令即可杀掉进程
```shell
kill -9 9608
```


## 9. 如何将 ubuntu 升级到最新版本  

推荐升级到最新的 22.04  

步骤

1、更新已安装的软件

```shell 
sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get dist-upgrade
```

2、移除不需要的软件

```shell 
sudo apt-get autoremove
```

3、安装更新管理

```shell 
sudo apt-get install update-manager-core
```

4、开始更新系统

```shell 
sudo do-release-upgrade
```

基本就是输入完命令，选择一些选项就可以完成升级了，基本不会遇到什么问题。

如果你在升级过程中遇到了“Checking for a new Ubuntu release Please install all available updates for your release before upgrading.”的问题，那么就再执行下下面的更新命令再升级系统：

```shell 
sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get dist-upgrade
```

5、检查系统版本

待更新完成后，使用命令lsb_release -a查看VPS的系统版本，看看是否成功升级成了Ubuntu 22.04：


## 10. 使用conda创建指定Python版本的虚拟环境

```shell 
conda create -n r2venv python=3.11
```


## 11. 如何卸载冲突的驱动

```shell 
yum remove xorg-x11-drv-nvidia\* kmod-nvidia\*
```

## 12. 如何让训练更快

1. 增大 logging_step ，避免频繁的记录 checkpoint 
2. 训练失败时，使用 trainer.train(resume_from_checkpoint=True) 从检查点恢复训练




