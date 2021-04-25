# 非结构化稀疏 -- 静态图剪裁（包括按照阈值和比例剪裁两种模式）

## 简介

在模型压缩中，常见的稀疏方式为结构化和非结构化稀疏，前者在某个特定维度（特征通道、卷积核等等）上进行稀疏化操作；后者以每一个参数为单元进行稀疏化，所以更加依赖于硬件对稀疏后矩阵运算的加速能力。本目录即在PaddlePaddle和PaddleSlim框架下开发的非结构化稀疏算法，MobileNetV1在ImageNet上的稀疏化实验中，剪裁率55.19%，达到无损的表现。

## 版本要求
```bash
python3.5+
paddlepaddle>=2.0.0
paddleslim>=2.1.0
```

请参照github安装[paddlepaddle](https://github.com/PaddlePaddle/Paddle)和[paddleslim](https://github.com/PaddlePaddle/PaddleSlim)。

## 使用

训练前：
- 预训练模型下载，并放到某目录下，通过train.py中的--pretrained_model设置。
- 训练数据下载后，可以通过重写../imagenet_reader.py文件，并在train.py文件中调用实现。
- 开发者可以通过重写paddleslim.prune.unstructured_pruner.py中的UnstructuredPruner.update_threshold()来定义自己的非结构化稀疏策略（目前为剪裁掉绝对值小的parameters）。
- 开发可以在初始化UnstructuredPruner时，传入自定义的skip_params_func，来定义哪些参数不参与剪裁。skip_params_func示例代码如下(路径：paddleslim.prune.unstructured_pruner._get_skip_params())。默认为所有的归一化层的参数不参与剪裁。

```python
def _get_skip_params(program):
    """
    The function is used to get a set of all the skipped parameters when performing pruning.
    By default, the normalization-related ones will not be pruned.
    Developers could replace it by passing their own function when initializing the UnstructuredPruner instance.
    Args:
      - program(paddle.static.Program): the current model.
    Returns:
      - skip_params(Set<String>): a set of parameters' names.
    """
    skip_params = set()
    graph = paddleslim.core.GraphWrapper(program)
    for op in graph.ops():
        if 'norm' in op.type() and 'grad' not in op.type():
            for input in op.all_inputs():
                skip_params.add(input.name())
    return skip_params
```

训练：
```bash
CUDA_VISIBLE_DEVICES=2,3 python3.7 train.py --data mnist --lr 0.1 --pruning_mode ratio --ratio=0.5
```

推理：
```bash
CUDA_VISIBLE_DEVICES=0 python3.7 evaluate.py --pruned_model models/ --data imagenet
```

剪裁训练代码示例：
```python
# model definition
places = paddle.static.cuda_places()
place = places[0]
exe = paddle.static.Executor(place)
model = models.__dict__[args.model]()
out = model.net(input=image, class_dim=class_dim)
cost = paddle.nn.functional.loss.cross_entropy(input=out, label=label)
avg_cost = paddle.mean(x=cost)
acc_top1 = paddle.metric.accuracy(input=out, label=label, k=1)
acc_top5 = paddle.metric.accuracy(input=out, label=label, k=5)

val_program = paddle.static.default_main_program().clone(for_test=True)

opt, learning_rate = create_optimizer(args, step_per_epoch)
opt.minimize(avg_cost)

#STEP1: initialize the pruner
pruner = UnstructuredPruner(paddle.static.default_main_program(), mode='ratio', ratio=0.5, place=place)

exe.run(paddle.static.default_startup_program())
paddle.fluid.io.load_vars(exe, args.pretrained_model)

for epoch in range(epochs):
    for batch_id, data in enumerate(train_loader):
        loss_n, acc_top1_n, acc_top5_n = exe.run(
            train_program,
            feed={
                "image": data[0].get('image'),
                "label": data[0].get('label')
            },
            fetch_list=[avg_cost.name, acc_top1.name, acc_top5.name])  
        learning_rate.step()
        #STEP2: update the pruner's threshold given the updated parameters
        pruner.step()

    if epoch % args.test_period == 0:
        #STEP3: before evaluation during training, eliminate the non-zeros generated by opt.step(), which, however, the cached masks setting to be zeros.
        pruner.update_params()
        eval(epoch)

    if epoch % args.model_period == 0:
        # STEP4: same purpose as STEP3
        pruner.update_params()
        save(epoch)
```

剪裁后测试代码示例：
```python
# intialize the model instance in static mode
# load weights
print(UnstructuredPruner.total_sparse(paddle.static.default_main_program())) #注意，total_sparse为静态方法(static method)，可以不创建实例(instance)直接调用，方便只做测试的写法。
test()
```

更多使用参数请参照shell文件，或者通过运行以下命令查看：
```bash
python3.7 train.py --h
python3.7 evaluate.py --h
```

## 实验结果

| 模型 | 数据集 | 压缩方法 | 压缩率| Top-1/Top-5 Acc | lr | threshold | epoch |
|:--:|:---:|:--:|:--:|:--:|:--:|:--:|:--:|
| MobileNetV1 | ImageNet | Baseline | - | 70.99%/89.68% | - | - | - |
| MobileNetV1 | ImageNet |   ratio  | -55.19% | 70.87%/89.80% (-0.12%/+0.12%) | 0.005 | - | 68 |
| YOLO v3     |  VOC     | - | - |76.24% | - | - | - |
| YOLO v3     |  VOC     |threshold | -55.15% | 75.45%(-0.79%) | 0.005 | 0.05 |12.8w|

## TODO

- [ ] 完成实验，验证动态图下的效果，并得到压缩模型。
- [ ] 扩充衡量parameter重要性的方法（目前仅为绝对值）。