import torch.optim as optim
import time
import xlwt
from datetime import datetime
from pathlib import Path

from src.dataset.data_loader import GMDataset, get_dataloader
from src.displacement_layer import Displacement
from src.loss_func import *
from src.evaluation_metric import matching_accuracy
from src.parallel import DataParallel
from src.utils.model_sl import load_model, save_model
from eval import eval_model
from src.lap_solvers.hungarian import hungarian
from src.utils.data_to_cuda import data_to_cuda

from src.utils.config import cfg
from pygmtools.benchmark import Benchmark


def train_eval_model(model, criterion, optimizer, dataloader, benchmark, num_epochs=25, start_epoch=0, xls_wb=None):
    print('Start training...')
    since = time.time()

    device = next(model.parameters()).device
    print('model on device: {}'.format(device))

    checkpoint_path = Path(cfg.OUTPUT_PATH) / 'params'
    if not checkpoint_path.exists():
        checkpoint_path.mkdir(parents=True)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.TRAIN.LR_STEP, gamma=cfg.TRAIN.LR_DECAY, last_epoch=cfg.TRAIN.START_EPOCH - 1)
    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        model.train()  # Set model to training mode
        print('lr = ' + ', '.join(['{:.2e}'.format(x['lr']) for x in optimizer.param_groups]))

        epoch_loss = 0.0
        running_loss = 0.0
        running_since = time.time()
        iter_num = 0

        # Iterate over data.
        while iter_num < cfg.TRAIN.EPOCH_ITERS:
            for inputs in dataloader['train']:
                if iter_num >= cfg.TRAIN.EPOCH_ITERS:
                    break
                if model.module.device != torch.device('cpu'):
                    inputs = data_to_cuda(inputs)

                iter_num = iter_num + 1
                # zero the parameter gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    # forward
                    outputs = model(inputs)
                    assert 'ds_mat' in outputs
                    assert 'perm_mat' in outputs
                    assert 'gt_perm_mat' in outputs
                    # compute loss
                    loss = criterion(outputs['ds_mat'], outputs['gt_perm_mat'], *outputs['ns'])
                    # compute accuracy
                    acc = matching_accuracy(outputs['perm_mat'], outputs['gt_perm_mat'], outputs['ns'][0])

                    # backward + optimize
                    loss.backward()
                    optimizer.step()

                    batch_num = inputs['batch_size']

                    loss_dict = dict()
                    loss_dict['loss'] = loss.item()
                    accdict = dict()
                    accdict['matching accuracy'] = torch.mean(acc)

                    # statistics
                    running_loss += loss.item() * batch_num
                    epoch_loss += loss.item() * batch_num

                    if iter_num % cfg.STATISTIC_STEP == 0:
                        running_speed = cfg.STATISTIC_STEP * batch_num / (time.time() - running_since)
                        print('Epoch {:<4} Iteration {:<4} {:>4.2f}sample/s Loss={:<8.4f}'
                              .format(epoch, iter_num, running_speed, running_loss / cfg.STATISTIC_STEP / batch_num))

                        running_loss = 0.0
                        running_since = time.time()

        epoch_loss = epoch_loss / cfg.TRAIN.EPOCH_ITERS / batch_num

        save_model(model, str(checkpoint_path / 'params_{:04}.pt'.format(epoch + 1)))
        torch.save(optimizer.state_dict(), str(checkpoint_path / 'optim_{:04}.pt'.format(epoch + 1)))

        print('Epoch {:<4} Loss: {:.4f}'.format(epoch, epoch_loss))
        print()

        # Eval in each epoch
        if dataloader['test'].dataset.clz not in ['none', 'all', None]:
            clss = [dataloader['test'].dataset.clz]
        else:
            clss = dataloader['test'].dataset.bm.classes
        l_e = (epoch == (num_epochs - 1))
        accs = eval_model(model, clss, benchmark['test'], l_e, xls_sheet=xls_wb.add_sheet('epoch{}'.format(epoch + 1)))
        acc_dict = {"{}".format(cls): single_acc for cls, single_acc in zip(dataloader['test'].dataset.classes, accs)}
        acc_dict['average'] = torch.mean(accs)
        wb.save(wb.__save_path)

        scheduler.step()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'
          .format(time_elapsed // 3600, (time_elapsed // 60) % 60, time_elapsed % 60))

    return model


if __name__ == '__main__':
    from src.utils.dup_stdout_manager import DupStdoutFileManager
    from src.utils.parse_args import parse_args
    from src.utils.print_easydict import print_easydict

    args = parse_args('Deep learning of graph matching training & evaluation code.')

    import importlib

    mod = importlib.import_module(cfg.MODULE)
    Net = mod.Net

    torch.manual_seed(cfg.RANDOM_SEED)

    dataset_len = {'train': cfg.TRAIN.EPOCH_ITERS * cfg.BATCH_SIZE, 'test': cfg.EVAL.SAMPLES}
    ds_dict = cfg[cfg.DATASET_FULL_NAME] if ('DATASET_FULL_NAME' in cfg) and (cfg.DATASET_FULL_NAME in cfg) else {}
    benchmark = {x: Benchmark(name=cfg.DATASET_FULL_NAME, sets=x, problem=cfg.PROBLEM.TYPE,
                    obj_resize=cfg.PROBLEM.RESCALE, filter=cfg.PROBLEM.FILTER, **ds_dict)
        for x in ('train', 'test')}

    train_dataset = GMDataset(name=cfg.DATASET_FULL_NAME,  bm=benchmark['train'], problem=cfg.PROBLEM.TYPE, length=dataset_len['train'], 
                              clz=cfg.TRAIN.CLASS, using_all_graphs=cfg.PROBLEM.TRAIN_ALL_GRAPHS)
    test_dataset = GMDataset(name=cfg.DATASET_FULL_NAME, bm=benchmark['test'], problem=cfg.PROBLEM.TYPE, length=dataset_len['test'], 
                             clz=cfg.EVAL.CLASS, using_all_graphs=cfg.PROBLEM.TEST_ALL_GRAPHS)

    image_dataset = {'train': train_dataset, 'test': test_dataset}
    dataloader = {x: get_dataloader(image_dataset[x], shuffle=True, fix_seed=(x == 'test')) for x in ('train', 'test')}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Net()
    model = model.to(device)

    criterion = PermutationLoss()

    if cfg.TRAIN.SEPARATE_BACKBONE_LR:
        backbone_ids = [id(item) for item in model.backbone_params]
        other_params = [param for param in model.parameters() if id(param) not in backbone_ids]

        model_params = [
            {'params': other_params},
            {'params': model.backbone_params, 'lr': cfg.TRAIN.BACKBONE_LR}
        ]
    else:
        model_params = model.parameters()

    if cfg.TRAIN.OPTIMIZER.lower() == 'sgd':
        optimizer = optim.SGD(model_params, lr=cfg.TRAIN.LR, momentum=cfg.TRAIN.MOMENTUM, nesterov=True)
    elif cfg.TRAIN.OPTIMIZER.lower() == 'adam':
        optimizer = optim.Adam(model_params, lr=cfg.TRAIN.LR)
    else:
        raise ValueError('Unknown optimizer {}'.format(cfg.TRAIN.OPTIMIZER))

    # model = DataParallel(model, device_ids=cfg.GPUS)

    if not Path(cfg.OUTPUT_PATH).exists():
        Path(cfg.OUTPUT_PATH).mkdir(parents=True)

    now_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    #tfboardwriter = SummaryWriter(logdir=str(Path(cfg.OUTPUT_PATH) / 'tensorboard' / 'training_{}'.format(now_time)))
    wb = xlwt.Workbook()
    wb.__save_path = str(Path(cfg.OUTPUT_PATH) / ('train_eval_result_' + now_time + '.xls'))

    with DupStdoutFileManager(str(Path(cfg.OUTPUT_PATH) / ('train_log_' + now_time + '.log'))) as _:
        print_easydict(cfg)
        model = train_eval_model(model, criterion, optimizer, dataloader,
                                 benchmark, num_epochs=cfg.TRAIN.NUM_EPOCHS, start_epoch=cfg.TRAIN.START_EPOCH,
                                 xls_wb=wb)

    wb.save(wb.__save_path)
