from lib.config import cfg, args
from lib.networks import make_network
from lib.train import make_trainer, make_optimizer, make_lr_scheduler, make_recorder, set_lr_scheduler
from lib.datasets.make_dataset_sxy import make_data_loader_sxy
from lib.utils.net_utils import load_model, save_model, load_network
from lib.evaluators.make_evaluator_sxy import make_evaluator_sxy
import torch.multiprocessing

# save_ep = 50
save_ep = 1
eval_ep = 20


mo_path = {
'roomTrain':'data/model/room_contour',
'roomVal':'data/model/room_contour'
}
if cfg.train.dataset in mo_path:
    cfg.model_dir = mo_path[cfg.train.dataset]

def train(cfg, network):
    trainer = make_trainer(cfg, network)
    optimizer = make_optimizer(cfg, network)
    scheduler = make_lr_scheduler(cfg, optimizer)
    recorder = make_recorder(cfg)
    evaluator = make_evaluator_sxy(cfg)

    begin_epoch = load_model(network, optimizer, scheduler, recorder, cfg.model_dir, resume=cfg.resume)
    # set_lr_scheduler(cfg, scheduler)

    train_loader = make_data_loader_sxy(cfg, is_train=True)
    val_loader = make_data_loader_sxy(cfg, is_train=False)


    print("\033[93m" + "batch_size {}---scatter to {}".format(cfg.train.batch_size,cfg.train.batch_size/3.0) + "\033[0m")
    print("\033[93m" + "GPUs {}; output_device {}".format(trainer.network.device_ids, trainer.network.output_device) + "\033[0m")
    # print("***test1***") #test_sxy

    print(len(train_loader))

    for epoch in range(begin_epoch, cfg.train.epoch):
        recorder.epoch = epoch
        # print("***test2***") #test_sxy
        trainer.train(epoch, train_loader, optimizer, recorder)
        # print("***test3***") #test_sxy
        scheduler.step()
        # print("***test4***") #test_sxy
        # # original version
        # if (epoch + 1) % cfg.save_ep == 0:
        if (epoch + 1) % save_ep == 0:
            save_model(network, optimizer, scheduler, recorder, epoch, cfg.model_dir)
        # # original version
        # if (epoch + 1) % eval_ep == 0:
            # trainer.val(epoch, val_loader, evaluator, recorder)
        if (epoch + 1) % eval_ep == 0:
            print("epoch: ", epoch+1)

    return network


def test(cfg, network):
    trainer = make_trainer(cfg, network)
    val_loader = make_data_loader_sxy(cfg, is_train=False)
    evaluator = make_evaluator_sxy(cfg)
    epoch = load_network(network, cfg.model_dir, resume=cfg.resume, epoch=cfg.test.epoch)
    trainer.val(epoch, val_loader, evaluator)


def main():
    network = make_network(cfg)
    if args.test:
        test(cfg, network)
    else:
        train(cfg, network)


if __name__ == "__main__":
    main()
