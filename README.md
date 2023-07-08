# MultiModel

本项目完成的任务是多模态情感分析。对于给定的图片和文本，分析其情感(positive, neutral, negative)。

多模态混合模型需要混合输入图片和文本数据，对其信息进行整合，然后输出其情感标签。因此，该任务是一个多模态输入数据的多分类任务。

# 代码环境

环境配置在[requirements.txt](requirements.txt)中列出。

* google_trans_new\=\=1.1.9
* langid=\=1.1.6
* Pillow=\=10.0.0
* torch=\=2.0.1+cu118
* torchvision=\=0.15.2+cu118
* transformers=\=4.24.0

可以运行下列命令安装环境依赖：

```sh
pip install -r requirements.txt
```

# 项目结构

```txt
├─codes---------------------------项目代码目录
│  │      main.py---------------------项目运行主程序
│  │
│  ├─models---------------------------模型代码目录
│  │      BertClassifier.py---------------BERT多分类模型
│  │      MobileNetV2.py------------------MobileNetV2图像处理模型
│  │      MultiModel.py-------------------多模态混合模型
│  │      state_dict_model.pt-------------保存的模型训练结果(已删除)
│  │
│  └─utils----------------------------工具文件夹
│         label_map.py-------------------将标签做映射的map
│         loadData.py--------------------数据预处理和导入数据的代码
│         trainer.py---------------------训练器代码
│
├─data----------------------------数据文件夹
```

# 数据集

由于数据集文件过大，在仓库中已全部删除(包括保存的模型训练结果)。如有需要可以联系:

10205501417@stu.ecnu.edu.cn

# 运行

在根目录，运行下列命令：

```python
python codes/main.py
```

或者直接运行脚本：

```shell
./run.sh
```

即可复现实验结果，展示模型消融下三种方式的验证集结果。

```txt
Validating...
valid acc 52.749996%
valid acc without fig 43.250000%
valid acc without txt 51.750000%
```

# 参考资料

[[1] MobileNetV2: Inverted Residuals and Linear Bottlenecks (arxiv.org)](https://arxiv.org/abs/1801.04381)

[[2]BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)

[[3]Issues · lushan88a/google_trans_new (github.com)](https://github.com/lushan88a/google_trans_new/issues)