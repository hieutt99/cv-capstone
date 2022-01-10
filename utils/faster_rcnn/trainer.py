import os, sys 
from utils.trainer import Trainer
import torch 
from tqdm.auto import tqdm 
from torchvision.utils import save_image 
from utils.metrics import *
import logging
logger = logging.getLogger(__name__)

class FasterRCNNTrainer(Trainer):
    def __init__(self, model , train_args, **kwargs):
        super(FasterRCNNTrainer, self).__init__(model=model, criterion=None, train_args=train_args)

    def compute_metrics(self):
        pass 

    def _convert_boxes(self, boxes, w, h):
        '''
        convert yolo boxes to xyxy boxes
        '''
        batch_size = boxes.size(0)
        pos_w = [[0,2] for i in range(batch_size)]
        pos_h = [[1,3] for i in range(batch_size)]
        w_components = torch.gather(boxes, dim=1, index=torch.tensor(pos_w))*w
        h_components = torch.gather(boxes, dim=1, index=torch.tensor(pos_h))*h
        boxes[:,0] = w_components[:, 0] - w_components[:,1]/2
        boxes[:,1] = h_components[:, 0] - h_components[:,1]/2
        boxes[:,2] = w_components[:, 0] + w_components[:,1]
        boxes[:,3] = h_components[:, 0] + h_components[:,1]
        return boxes

    def _handle_batch(self, batch):
        imgs, labels, paths, _ = batch 
        b, c, w, h = imgs.size()
        imgs = [img/255 for img in imgs]
        targets = []
        for i in range(len(imgs)):
            query = labels[(labels[:,0]==i).nonzero().squeeze(1)]
            _labels = query[:,1].long()
            _boxes = query[:,-4:]
            targets.append({
                'boxes': self._convert_boxes(_boxes, w, h) if _boxes.size(0)>0 else _boxes,
                'labels': _labels,
            })
        return imgs, targets    

    def _train_one_batch(self, batch):
        self.model.train()
        imgs, targets = self._handle_batch(batch)

        outputs = self.model(imgs, targets)

        return outputs 

    def _eval_one_batch(self, batch):
        self.model.eval()
        imgs, targets = self._handle_batch(batch)

        predictions = self.model(imgs)

        metrics = None 
        if self.compute_metrics:
            metrics = self.compute_metrics(predictions, targets)
        return predictions, metrics 

    def eval(self, loader):
        num_iter = len(loader)
        progress_bar = tqdm(range(num_iter))
        progress_bar.set_description("Eval: ")
        for index, batch in enumerate(loader):
            predictions, metrics = self._eval_one_batch(batch)

            progress_bar.update(1)
        progress_bar.close()
        return 

    def _train_with_epoch(self, loader):
        for epoch in range(self.args.n_epochs):
            iter_num = len(loader)
            progress_bar = tqdm(range(iter_num))
            progress_bar.set_description(f"Training epoch {epoch}: ")
            n_steps = len(loader)
            if self.args.steps_per_epoch:
                n_steps = self.args.steps_per_epoch
            iterator = iter(loader)
            for step in range(n_steps):
                try:
                    batch = next(iterator)
                except:
                    iterator = iter(loader)
                    batch = next(iterator)

                outputs = self._train_one_batch(batch)
                
                progress_bar.update(1)
                progress_bar.set_postfix(outputs)
                
                for k, v in outputs.items():
                    self.writer.add_scalar(k, v.item(), global_step=self.global_step)
                self.global_step+=1
                
                if self.global_step%self.args.lr_steps == 0:
                    self.lr_scheduler.step()

                if self.global_step%self.args.save_interval == 0 and self.args.save_strategy=='step':
                    self._save_checkpoint()
                    self.save_trainer_state()
            progress_bar.close()
            self.global_epoch += 1
            if self.global_epoch%self.args.save_interval == 0 and self.args.save_strategy == 'epoch':
                self._save_checkpoint()
                self.save_trainer_state()

        self._save_checkpoint()
        self.save_trainer_state()
