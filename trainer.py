from comet_ml import Experiment
import torch
import time
import numpy as np
from torchvision.utils import make_grid
from torchvision import transforms
from torch.autograd import Variable
from torch import nn
import PIL
import csv
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score
import seaborn as sns


from utils import transforms as local_transforms
from base import BaseTrainer, DataPrefetcher
from utils.helpers import colorize_mask, confusion_matrix_Kit
from utils.metrics import eval_metrics, AverageMeter
from tqdm.auto import tqdm






class Trainer(BaseTrainer):
    def __init__(self, model, loss, resume, config,
                 train_loader, val_loader=None,
                 train_logger=None, prefetch=True):
        super(Trainer, self).__init__(
                model, loss, resume, config,
                train_loader, val_loader, train_logger)
        
        self.wrt_mode, self.wrt_step = 'train_', 0
        self.log_step = config['trainer'].get(
                'log_per_iter', int(np.sqrt(self.train_loader.batch_size)))
        if config['trainer']['log_per_iter']:
            self.log_step = int(
                    self.log_step / self.train_loader.batch_size) + 1

        self.num_classes = self.train_loader.dataset.num_classes


        
        # TRANSORMS FOR VISUALIZATION
        if self.train_loader.MEAN[0]==None:
            self.restore_transform = transforms.Compose([
                transforms.ToPILImage()])
        else:
            self.restore_transform = transforms.Compose([
                local_transforms.DeNormalize(
                        self.train_loader.MEAN, self.train_loader.STD),
                transforms.ToPILImage()])

        
        self.restore_toPill= transforms.Compose([transforms.ToPILImage()])

        self.viz_transform = transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.ToTensor()])

        if self.device == torch.device('cpu'):
            prefetch = False
        if prefetch:
            self.train_loader = DataPrefetcher(
                    train_loader, device=self.device)
            self.val_loader = DataPrefetcher(
                    val_loader, device=self.device)
            
        torch.backends.cudnn.benchmark = True
        

    def _train_epoch(self, epoch):
        
        self.logger.info('\n')

        self.model.train()
        if self.config['arch']['args']['freeze_bn']:
            if isinstance(self.model, torch.nn.DataParallel):
                self.model.module.freeze_bn()
            else:
                self.model.freeze_bn()

        self.wrt_mode = 'train'
        tic = time.time()
        self._reset_metrics()
        pbar = tqdm(self.train_loader)
         
        
        for batch_idx, (data, target) in enumerate(pbar):
            self.data_time.update(time.time() - tic)
            # data, target = data.to(self.device), target.to(self.device)
            #self.lr_scheduler.step(epoch=epoch-1)

            # LOSS & OPTIMIZE
            self.optimizer.zero_grad()
            output = self.model(data)
            if self.config['arch']['type'][:3] == 'PSP':
                assert output[0].size()[2:] == target.size()[1:]
                assert output[0].size()[1] == self.num_classes
                loss = self.loss(output[0], target)
                loss += self.loss(output[1], target) * 0.4
                output = output[0]
            else:
                assert output.size()[2:] == target.size()[1:]
                assert output.size()[1] == self.num_classes
                loss = self.loss(output, target)

            if isinstance(self.loss, torch.nn.DataParallel):    
                loss = loss.mean()
            loss.backward()     
            self.optimizer.step()
            
            # Note updated as per pytorch UserWarning: The epoch parameter in `scheduler.step()` was not necessary
            self.lr_scheduler.step(epoch=epoch-1)
            self.total_loss.update(loss.item())
            
            # measure elapsed time
            self.batch_time.update(time.time() - tic)
            tic = time.time()

            # LOGGING & TENSORBOARD
            if batch_idx % self.log_step == 0:
                
                #TODo: Stability check, Debug the graph using tensorboardx 
                #dummy_shape_input = Variable(torch.randn(8, 3,400, 400, device='cuda'))
                #self.writer.add_graph(self.model, dummy_shape_input, True)
                
                self.wrt_step = (epoch - 1) * len(
                        self.train_loader) + batch_idx
                self.writer.add_scalar(
                        f'{self.wrt_mode}/loss', loss.item(), self.wrt_step)

            # FOR EVAL
            seg_metrics = eval_metrics(output, target, self.num_classes)
            self._update_seg_metrics(*seg_metrics)
            pixAcc, mIoU, _ = self._get_seg_metrics().values()

            # PRINT INFO
            pbar.set_description(
                    "TRAIN: {} | Loss {:.2f} | Acc {:.2f} mIoU {:.2f}| B {:.2f} D {:.2f} |".format(
                                                epoch, self.total_loss.average,
                                                pixAcc, mIoU,
                                                self.batch_time.average,
                                                self.data_time.average))
            
        
        pbar.close()
        
        # METRICS TO TENSORBOARD
        seg_metrics = self._get_seg_metrics()
        

        # RETURN LOSS & METRICS
        log = {'loss': self.total_loss.average, **seg_metrics}

        if self.lr_scheduler is not None: 
            self.lr_scheduler.step()
            
        return log

    def _valid_epoch(self, epoch):
            if self.val_loader is None:
                self.logger.warning(
                        """Not data loader was passed for the validation step,
                        No validation is performed !""")
                return {}
            self.logger.info('\n###### EVALUATION ######')

            self.model.eval()
            self.wrt_mode = 'val'

            self._reset_metrics()
            tbar = tqdm(self.val_loader)
            with torch.no_grad():
                val_visual = []
                for batch_idx, (data, target) in enumerate(tbar):
                    # data, target = data.to(self.device), target.to(self.device)
                    # LOSS
                    output = self.model(data)
                    loss = self.loss(output, target)
                    if isinstance(self.loss, torch.nn.DataParallel):
                        loss = loss.mean()
                    self.total_loss.update(loss.item())

                    seg_metrics = eval_metrics(output, target, self.num_classes)
                    self._update_seg_metrics(*seg_metrics)

                    # LIST OF IMAGE TO VIZ (15 images)
                    if len(val_visual) < 15:
                        target_np = target.data.cpu().numpy()
                        output_np = output.data.max(1)[1].cpu().numpy()
                        val_visual.append([data[0].data.cpu(), target_np[0], output_np[0]])

                    # PRINT INFO
                    pixAcc, mIoU, _ = self._get_seg_metrics().values()
                    tbar.set_description(
                            """EVAL ({}) | Loss: {:.3f},
                            PixelAcc: {:.2f}, Mean IoU: {:.2f} |"""
                            .format(epoch, self.total_loss.average, pixAcc, mIoU))

                # WRTING & VISUALIZING THE MASKS
                val_img = []
                palette = self.train_loader.dataset.palette
                for d, t, o in val_visual:
                    d = self.restore_transform(d)
                    t, o = colorize_mask(t, palette), colorize_mask(o, palette)
                    d, t, o = d.convert('RGB'), t.convert('RGB'), o.convert('RGB')
                    [d, t, o] = [self.viz_transform(x) for x in [d, t, o]]
                    val_img.extend([d, t, o])
                val_img = torch.stack(val_img, 0)
                val_img = make_grid(val_img.cpu(), nrow=3, padding=5)

                #self.writer.add_image(
                #        f'{self.wrt_mode}/inputs_targets_predictions',
                #       val_img, self.wrt_step)

                # METRICS TO TENSORBOARD
                self.wrt_step = (epoch) * len(self.val_loader)
            # self.writer.add_scalar(
                #        f'{self.wrt_mode}/loss',
                #        self.total_loss.average, self.wrt_step)

                seg_metrics = self._get_seg_metrics()
                # for k, v in list(seg_metrics.items())[:-1]:
                    # self.writer.add_scalar(
                            # f'{self.wrt_mode}/{k}', v, self.wrt_step)

                log = {
                    'val_loss': self.total_loss.average,
                    **seg_metrics
                }

            return log

    def _test_epoch(self, epoch):
        if self.val_loader is None:
            self.logger.warning(
                    """Not data loader was passed for the validation step,
                    No validation is performed !""")
            return {}
        self.logger.info('\n###### EVALUATION ######')

        self.model.eval()

        self.wrt_mode = 'val'

        self._reset_metrics()
        tbar = tqdm(self.val_loader)
        Counter = 0

        # ToDo: Code as in test epoch
        # dir_Write = "/media/freddy/vault/datasets/Grassclover/eval_results"
        dir_Write = "/media/freddy/vault/datasets/greenway/all/test/Divot_SimonV2/Images_Processed/Results"
        file = open(dir_Write+'/result.csv', 'w')
        writer = csv.writer(file)
        # correct, labeled, inter, union
        header = ["Image ID",'correct', 'labeled']
        for i in range(self.num_classes):
            header.extend('inter')
        for i in range(self.num_classes):
            header.extend('union')      

        writer.writerow(header)

        with torch.no_grad():
            val_visual = []
            for batch_idx, (data, target) in enumerate(tbar):
                data, target = data.to(self.device), target.to(self.device)
                # LOSS
                output = self.model(data)
                loss = self.loss(output, target)
                if isinstance(self.loss, torch.nn.DataParallel):
                    loss = loss.mean()
                self.total_loss.update(loss.item())

                
                
                #print("Unique IDS before: ", np.unique(label, return_counts=True))
                # Remapping here

                '''
                binary_Class_remap = {0:0, 
                1: 0, 
                2: 0, 
                3: 0,
                4: 1,
                5: 0, 
                6: 0}

                _, output2 = torch.max(output.data, 1)

                for k, v in binary_Class_remap.items():
                    output2[output2 == k] = v

                for k, v in binary_Class_remap.items():
                    target[target == k] = v
                '''
                seg_metrics = eval_metrics(output, target, self.num_classes)
                self._update_seg_metrics(*seg_metrics)

                
                correct_img, labeled_img, inter_img, union_img = seg_metrics
                pixAcc_img = 1.0 * correct_img / (np.spacing(1) + labeled_img)
                IoU_img = 1.0 * inter_img / (np.spacing(1) + union_img)
                mIoU_img = IoU_img.mean()

                # Create a name for the log file
                imageName = "BatchID"+str(batch_idx)
                data_metric = [imageName ,seg_metrics[0], seg_metrics[1]]
                data_metric.extend(seg_metrics[2])
                data_metric.extend(seg_metrics[3])
                writer.writerow(data_metric)


                # Visualize first 20 images todo put this pass as eval parameter
                # Code for images for publications 
                # Class_IoU      : {0: 0.09725328, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.37005183, 5: 0.0}
                # Visualize issus only 

                if Counter <= 20:
                #if IoU_img[1] < 0.2 and IoU_img[1]!= 0: 

                    blockOfImage = 4 
                    if len(val_visual) < blockOfImage:
                        input_np = data[0].data.cpu()        
                        target_np = target.data.cpu().numpy()                  
                        output_np = output.data.max(1)[1].cpu().numpy()
                        val_visual.append([input_np , target_np[0], output_np[0]])

                    else:
                        # WRTING & VISUALIZING THE MASKS
                        val_img = []
                        palette = self.train_loader.dataset.palette
                        for data_raw, target_colour_mask, output_colour_mask in val_visual:
                            
                            #Todo: Add to config the mean on and off
                            # if self.config:
                            # data_raw = self.restore_transform(data_raw)
                            data_raw = self.restore_toPill(data_raw)


                            # ref : https://developmentseed.org/tensorflow-eo-training/docs/Lesson4_evaluation.html
                            flat_preds = np.concatenate(target_colour_mask).flatten()
                            flat_truth = np.concatenate(output_colour_mask).flatten()
                            
                            #Name for the classes in the confusion matrix
                            dataset_baseline=  [ 'grass', 'white clover', 'red clover', 'weeds', 'soil', 'clover other']
                            dataset_baseline=  [ 'veg', 'soil', ]
                            # dataset_baseline= [ 'grass', 'white clover', 'red clover', 'weeds', 'soil', 'clover other', 'boundary']
                            # cf_matrix = confusion_matrix(flat_truth, flat_preds )

                            # ax = sns.heatmap(cf_matrix , fmt='.2%', cmap='Blues')
                            # ax.set_title('Seaborn Confusion Matrix with labels\n\n')
                            # ax.set_xlabel('\nPredicted Flower Category')
                            # ax.set_ylabel('Actual Flower Category ')

                            # ## Ticket labels - List must be in alphabetical order
                            # ax.xaxis.set_ticklabels(dataset_clover_baseline)
                            # ax.yaxis.set_ticklabels(dataset_clover_baseline)

                            # ax.set_title('Seaborn Confusion Matrix with labels\n\n')
                            
                            # ref: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html
                            disp = ConfusionMatrixDisplay.from_predictions(flat_truth, flat_preds, normalize=None, labels=list(range(len(dataset_baseline))), display_labels = dataset_baseline, xticks_rotation=45)

                            ## Display the visualization of the Confusion Matrix.
                        
                            # Temp fix for moving between 
                            write_confusion_matrix = dir_Write+'/temp_confusion_matrix.png'
                            # ax.figure.savefig(write_confusion_matrix)
                            # ax.figure.clf()
                            disp.figure_.savefig(write_confusion_matrix)


                            # gen_makedRegion = data_raw.copy() 
                            # ref: https://www.kite.com/blog/python/image-segmentation-tutorial/
                            confusion_matrix_arrs = confusion_matrix_Kit(target_colour_mask, output_colour_mask )
                            color_mask = np.zeros_like(data_raw)
                            confusion_matrix_colors = {
                                # Confusion: (R,G,B)   
                                'tp': (255, 0, 255),  #Cyan Note:  True positive (TP): Observation is predicted positive and is actually positive
                                'fp': (0, 0, 255),  # blue False positive (FP): Observation is predicted positive and is actually negative
                                'fn': (255, 255, 0),  #yellow false negatives (FN): Observation predicted no, but they actually has the class
                                'tn': (0, 0, 0)     #black True negative (TN): Observation is predicted negative and is actually negative
                                }

                            for predict, mask in confusion_matrix_arrs.items():
                                color = confusion_matrix_colors[predict]
                                mask_rgb = np.zeros_like(data_raw)
                                mask_rgb[mask != 0] = color
                                color_mask += mask_rgb

                            
                            im_read = PIL.Image.open(write_confusion_matrix)
                            mywidth = 400
                            wpercent = (mywidth/float(im_read.size[0]))
                            hsize = int((float(im_read.size[1])*float(wpercent)))

                            vr_raw  = im_read.resize((mywidth, hsize), PIL.Image.ANTIALIAS)
                            # write_confusion_matrix = dir_Write+'/temp_confusion_matrixV2.png'
                            # vr_raw.save( write_confusion_matrix)


                            vr_raw = PIL.ImageOps.pad(vr_raw, (400, 400), method=PIL.Image.Resampling.BICUBIC, color=None, centering=(0.5, 0.5))
                            #im_read = PIL.ImageOps.fit(im_read, (400, 400), method=PIL.Image.Resampling.BICUBIC, bleed=0.0, centering=(0.5, 0.5))

                                                    # Call draw Method to add 2D graphics in an image
                            I1 = PIL.ImageDraw.Draw(vr_raw)
                            # Custom font style and font size
                            myFont = PIL.ImageFont.truetype('FreeMono.ttf', 12)

                            # Add Text to an image


                            strRav= "Pixel_Accuracy "+str(np.round(pixAcc_img, 2))+" Mean_IoU "+str(np.round(mIoU_img, 2))
                            strRav2= "C IoU "+str(dict(zip(range(self.num_classes), np.round(IoU_img,2) )))
                            # strRav= """ PixelAcc: {:.2f}, Mean IoU: {:.2f} |""".format(pixAcc, mIoU_img)
                            
                            I1.text((2, 2), strRav,font=myFont, fill=(255, 0, 0))
                            I1.text((2, 15), strRav2,font=myFont, fill=(255, 0, 0))
                            # Transform to Torch type
                            vr_raw = self.viz_transform(vr_raw.convert('RGB'))                          


                            target_colour_mask, output_colour_mask = colorize_mask(target_colour_mask, palette), colorize_mask(output_colour_mask, palette)
                            data_raw, target_colour_mask, output_colour_mask = data_raw.convert('RGB'), target_colour_mask.convert('RGB'), output_colour_mask.convert('RGB')
                            [data_raw, target_colour_mask, output_colour_mask] = [self.viz_transform(x) for x in [data_raw, target_colour_mask, output_colour_mask]]
                           
                            # Compare 3d colours interesting ouput
                            # vr = target_colour_mask- output_colour_mask
                            val_img.extend([data_raw, target_colour_mask, output_colour_mask, vr_raw])
                        
                        val_img = torch.stack(val_img, 0)
                        val_img = make_grid(val_img.cpu(), nrow=4, padding=5)
                        image_label = transforms.ToPILImage()(val_img )
                                                    


                        image_label.save(dir_Write+"/BatchID_"+str(batch_idx-blockOfImage)+str("_to_")+str(batch_idx)+".png")
                        val_visual.clear()

                # PRINT INFO
                self.wrt_step = (epoch) * len(self.val_loader)
                pixAcc, mIoU, _ = self._get_seg_metrics().values()
                tbar.set_description(
                        """EVAL ({}) | Loss: {:.3f},
                        PixelAcc: {:.2f}, Mean IoU: {:.2f} |"""
                        .format(epoch, self.total_loss.average, pixAcc, mIoU))
                
                Counter = Counter +1



                #if(Counter == 500):
                #    break
                
            

            # WRTING & VISUALIZING THE MASKS FOR PUBLICATIONS
            '''
            val_img = []
            palette = self.train_loader.dataset.palette
            for d, t, o in val_visual:
                d = self.restore_transform(d)
                t, o = colorize_mask(t, palette), colorize_mask(o, palette)
                d, t, o = d.convert('RGB'), t.convert('RGB'), o.convert('RGB')
                [d, t, o] = [self.viz_transform(x) for x in [d, t, o]]
                val_img.extend([d, t, o])
            val_img = torch.stack(val_img, 0)
            val_img = make_grid(val_img.cpu(), nrow=3, padding=5)
            image_label = transforms.ToPILImage()(val_img )
          
            image_label.save("/media/freddy/vault/datasets/Grassclover/colab_version/result_masks/test.png")


            self.writer.add_image(
                    f'{self.wrt_mode}/inputs_targets_predictions',
                    val_img, self.wrt_step)
            '''            
            

            #close the data file
            file.close()

            # METRICS TO TENSORBOARD
            self.wrt_step = (epoch) * len(self.val_loader)
            self.writer.add_scalar(
                                f'{self.wrt_mode}/loss',
                                self.total_loss.average, self.wrt_step)
            self.experiment.log_metric("accuracy", mIoU, step=self.wrt_step)
            

            seg_metrics = self._get_seg_metrics()
            for k, v in list(seg_metrics.items())[:-1]:
                self.writer.add_scalar(
                    f'{self.wrt_mode}/{k}', v, self.wrt_step)

            log = {
                'val_loss': self.total_loss.average,
                **seg_metrics
            }

        return log

    def _reset_metrics(self):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.total_loss = AverageMeter()
        self.total_inter, self.total_union = 0, 0
        self.total_correct, self.total_label = 0, 0

    def _update_seg_metrics(self, correct, labeled, inter, union):
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union

    def _get_seg_IOU_metrics(self):
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        return  IoU

    def _get_seg_metrics(self):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        
        return {
            "Pixel_Accuracy": np.round(pixAcc, 3),
            "Mean_IoU": np.round(mIoU, 3),
            "Class_IoU": dict(zip(range(self.num_classes), IoU))
        }

    
