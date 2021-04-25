import paddle
import os
import sys
import argparse
import numpy as np
from paddleslim.dygraph.prune.unstructured_pruner import UnstructuredPruner
sys.path.append(
    os.path.join(os.path.dirname("__file__"), os.path.pardir, os.path.pardir))
from utility import add_arguments, print_arguments
import paddle.vision.transforms as T
import paddle.nn.functional as F
import functools
from paddle.vision.models import mobilenet_v1
import time
import logging
from paddleslim.common import get_logger
import paddle.distributed as dist

_logger = get_logger(__name__, level=logging.INFO)

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('batch_size',       int,  64,                 "Minibatch size.")
add_arg('use_gpu',          bool, True,                "Whether to use GPU or not.")
add_arg('lr',               float,  0.1,               "The learning rate used to fine-tune pruned model.")
add_arg('lr_strategy',      str,  "piecewise_decay",   "The learning rate decay strategy.")
add_arg('l2_decay',         float,  3e-5,               "The l2_decay parameter.")
add_arg('momentum_rate',    float,  0.9,               "The value of momentum_rate.")
add_arg('ratio',            float,  0.3,               "The ratio to set zeros, the smaller part bounded by the ratio will be zeros.")
add_arg('pruning_mode',            str,  'ratio',               "the pruning mode: whether by ratio or by threshold.")
add_arg('threshold',            float,  0.001,               "The threshold to set zeros.")
add_arg('num_epochs',       int,  120,               "The number of total epochs.")
parser.add_argument('--step_epochs', nargs='+', type=int, default=[30, 60, 90], help="piecewise decay step")
add_arg('data',             str, "cifar10",                 "Which data to use. 'cifar10' or 'imagenet'.")
add_arg('log_period',       int, 100,                 "Log period in batches.")
add_arg('test_period',      int, 1,                 "Test period in epoches.")
add_arg('model_path',       str, "./models",         "The path to save model.")
add_arg('model_period',     int, 10,             "The period to save model in epochs.")
add_arg('resume_epoch',     int, -1,             "The epoch to resume training.")
add_arg('num_workers',     int, 4,             "number of workers when loading dataset.")
# yapf: enable


def piecewise_decay(args, step_per_epoch, model):
    bd = [step_per_epoch * e for e in args.step_epochs]
    lr = [args.lr * (0.1**i) for i in range(len(bd) + 1)]
    learning_rate = paddle.optimizer.lr.PiecewiseDecay(boundaries=bd, values=lr)

    optimizer = paddle.optimizer.Momentum(
        learning_rate=learning_rate,
        momentum=args.momentum_rate,
        weight_decay=paddle.regularizer.L2Decay(args.l2_decay),
        parameters=model.parameters())
    return optimizer, learning_rate


def cosine_decay(args, step_per_epoch, model):
    learning_rate = paddle.optimizer.lr.CosineAnnealingDecay(
        learning_rate=args.lr, T_max=args.num_epochs * step_per_epoch)
    optimizer = paddle.optimizer.Momentum(
        learning_rate=learning_rate,
        momentum=args.momentum_rate,
        weight_decay=paddle.regularizer.L2Decay(args.l2_decay),
        parameters=model.parameters())
    return optimizer, learning_rate


def create_optimizer(args, step_per_epoch, model):
    if args.lr_strategy == "piecewise_decay":
        return piecewise_decay(args, step_per_epoch, model)
    elif args.lr_strategy == "cosine_decay":
        return cosine_decay(args, step_per_epoch, model)


def compress(args):
    dist.init_parallel_env()
    train_reader = None
    test_reader = None
    if args.data == "imagenet":
        import imagenet_reader as reader
        train_dataset = reader.ImageNetDataset(data_dir='/data', mode='train')
        val_dataset = reader.ImageNetDataset(data_dir='/data', mode='val')
        class_dim = 1000
    elif args.data == "cifar10":
        normalize = T.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], data_format='CHW')
        transform = T.Compose([T.Transpose(), normalize])
        train_dataset = paddle.vision.datasets.Cifar10(
            mode='train', backend='cv2', transform=transform)
        val_dataset = paddle.vision.datasets.Cifar10(
            mode='test', backend='cv2', transform=transform)
        class_dim = 10
    else:
        raise ValueError("{} is not supported.".format(args.data))
    places = paddle.static.cuda_places(
    ) if args.use_gpu else paddle.static.cpu_places()
    batch_size_per_card = int(args.batch_size / len(places))
    train_loader = paddle.io.DataLoader(
        train_dataset,
        places=places,
        drop_last=True,
        batch_size=args.batch_size,
        shuffle=True,
        return_list=True,
        num_workers=args.num_workers,
        use_shared_memory=True)
    valid_loader = paddle.io.DataLoader(
        val_dataset,
        places=places,
        drop_last=False,
        return_list=True,
        batch_size=args.batch_size,
        shuffle=False,
        use_shared_memory=True)
    step_per_epoch = int(np.ceil(len(train_dataset) * 1. / args.batch_size))

    # model definition
    model = mobilenet_v1(num_classes=class_dim, pretrained=True)
    dp_model = paddle.DataParallel(model)

    opt, learning_rate = create_optimizer(args, step_per_epoch, dp_model)

    def test(epoch):
        dp_model.eval()
        acc_top1_ns = []
        acc_top5_ns = []
        for batch_id, data in enumerate(valid_loader):
            start_time = time.time()
            x_data = data[0]
            y_data = paddle.to_tensor(data[1])
            if args.data == 'cifar10':
                y_data = paddle.unsqueeze(y_data, 1)
            end_time = time.time()

            logits = dp_model(x_data)
            loss = F.cross_entropy(logits, y_data)
            acc_top1 = paddle.metric.accuracy(logits, y_data, k=1)
            acc_top5 = paddle.metric.accuracy(logits, y_data, k=5)

            acc_top1_ns.append(acc_top1.numpy())
            acc_top5_ns.append(acc_top5.numpy())
            if batch_id % args.log_period == 0:
                _logger.info(
                    "Eval epoch[{}] batch[{}] - acc_top1: {}; acc_top5: {}; time: {}".
                    format(epoch, batch_id,
                           np.mean(acc_top1.numpy()),
                           np.mean(acc_top5.numpy()), end_time - start_time))
            acc_top1_ns.append(np.mean(acc_top1.numpy()))
            acc_top5_ns.append(np.mean(acc_top5.numpy()))

        _logger.info("Final eval epoch[{}] - acc_top1: {}; acc_top5: {}".format(
            epoch,
            np.mean(np.array(
                acc_top1_ns, dtype="object")),
            np.mean(np.array(
                acc_top5_ns, dtype="object"))))

    def train(epoch):
        dp_model.train()
        for batch_id, data in enumerate(train_loader):
            start_time = time.time()
            x_data = data[0]
            y_data = paddle.to_tensor(data[1])
            if args.data == 'cifar10':
                y_data = paddle.unsqueeze(y_data, 1)

            logits = dp_model(x_data)
            loss = F.cross_entropy(logits, y_data)
            acc_top1 = paddle.metric.accuracy(logits, y_data, k=1)
            acc_top5 = paddle.metric.accuracy(logits, y_data, k=5)
            end_time = time.time()
            if batch_id % args.log_period == 0:
                _logger.info(
                    "epoch[{}]-batch[{}] lr: {:.6f} - loss: {}; acc_top1: {}; acc_top5: {}; time: {}".
                    format(epoch, batch_id, args.lr,
                           np.mean(loss.numpy()),
                           np.mean(acc_top1.numpy()),
                           np.mean(acc_top5.numpy()), end_time - start_time))
            loss.backward()
            opt.step()
            opt.clear_grad()
            pruner.step()

    pruner = UnstructuredPruner(
        dp_model,
        mode=args.pruning_mode,
        ratio=args.ratio,
        threshold=args.threshold)
    for i in range(args.resume_epoch + 1, args.num_epochs):
        train(i)
        if i % args.test_period == 0:
            pruner.update_params()
            _logger.info(
                "The current density of the pruned model is: {}%".format(
                    round(100 * UnstructuredPruner.total_sparse(dp_model), 2)))
            test(i)
        if i > args.resume_epoch and i % args.model_period == 0:
            pruner.update_params()
            paddle.save(dp_model.state_dict(),
                        os.path.join(args.model_path, "model-pruned.pdparams"))
            paddle.save(opt.state_dict(),
                        os.path.join(args.model_path, "opt-pruned.pdopt"))


def main():
    args = parser.parse_args()
    print_arguments(args)
    compress(args)


if __name__ == '__main__':
    main()