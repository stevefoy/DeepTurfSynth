import os
import json
import argparse
import torch
import dataloaders
import models
from utils import losses
from utils import LoggerFormat
from trainer import Trainer
# Import comet_ml at the top of your file

# Code topic @ https://stackoverflow.com/questions/17251008/python-call-a-constructor-whose-name-is-stored-in-a-variable
def get_instance(module, cls_name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT
    print("Loading class", config[cls_name]['type'])
    cls = getattr(module, config[cls_name]['type'])

            # load class(*args, **kwargs) 
    return cls(*args, **config[cls_name]['args'])


def main(config, args):
    train_logger = LoggerFormat()

    # DATA LOADERS
    train_loader = get_instance(dataloaders, 'train_loader', config)
    val_loader = get_instance(dataloaders, 'val_loader', config)
    try: 
        test_loader = get_instance(dataloaders, 'test_loader', config)
    except:
        print("No test_loader in config file")

    # MODEL
    model = get_instance(
            models, 'arch', config, train_loader.dataset.num_classes)
    print(f'\n{model}\n')

    # LOSS
    loss = getattr(losses, config['loss'])(
        ignore_index=config['ignore_index'], weight=config['weight'])

    # TRAINING
    trainer = Trainer(
        model=model,
        loss=loss,
        resume=args.resume,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        train_logger=train_logger)

    
    if args.eval:
        trainer.eval_weights()
    elif args.test:
        # raise NotImplementedError
        trainer.test_weights()
    else:
        trainer.train()

    

if __name__ == '__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='config.json', type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument('-e', '--eval', default=False, type=bool,
                        help='Run evaluation on weights')
    parser.add_argument('-t', '--test', default=False, type=bool,
                        help='Run test on weights')

    args = parser.parse_args()

    config = json.load(open(args.config))
    if args.resume:
        pass
        # Usually we don't want to resume from stored config in checkpoint 
        # config = torch.load(args.resume)['config']
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device


    main(config, args)
