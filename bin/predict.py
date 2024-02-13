#!/usr/bin/env python3

# Example command:
# ./bin/predict.py \
#       model.path=<path to checkpoint, prepared by make_checkpoint.py> \
#       indir=<path to input data> \
#       outdir=<where to store predicts>

import logging
import os
import sys
import traceback

from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.evaluation.refinement import refine_predict
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import cv2
import hydra
import numpy as np
import torch
import tqdm
import yaml
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate

from saicinpainting.training.data.datasets import make_default_val_dataset
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.utils import register_debug_signal_handlers

LOGGER = logging.getLogger(__name__)


@hydra.main(config_path='../configs/prediction', config_name='map.yaml')
def main(predict_config: OmegaConf):
    try:
        if sys.platform != 'win32':
            register_debug_signal_handlers()  # kill -10 <pid> will result in traceback dumped into log

        device = torch.device(predict_config.device)

        # Set up different models 
        model_list = []
        for model_path in predict_config.model.paths:
            train_config_path = os.path.join(model_path, 'config.yaml')
            with open(train_config_path, 'r') as f:
                train_config = OmegaConf.create(yaml.safe_load(f))
            
            train_config.training_model.predict_only = True
            train_config.visualizer.kind = 'noop'

            out_ext = predict_config.get('out_ext', '.png')

            checkpoint_path = os.path.join(model_path, 
                                        'models', 
                                        predict_config.model.checkpoint)
            model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
            model.freeze()
            if not predict_config.get('refine', False):
                model.to(device)
            model_list.append(model)

        if not predict_config.indir.endswith('/'):
            predict_config.indir += '/'

        dataset = make_default_val_dataset(predict_config.indir, **predict_config.dataset)
        
        for img_i in tqdm.trange(len(dataset)):
            mask_fname = dataset.obs_img_filenames[img_i]
            cur_out_fname = os.path.join(
                predict_config.outdir, 
                (os.path.splitext(mask_fname[len(predict_config.indir):])[0]).split('/')[-1] + out_ext
            )
            
            os.makedirs(os.path.dirname(cur_out_fname), exist_ok=True)
            batch = default_collate([dataset[img_i]])
            # import pdb; pdb.set_trace()
            pred_result = []
            for model_i, model in enumerate(model_list):
                if predict_config.get('refine', False):
                    assert 'unpad_to_size' in batch, "Unpadded size is required for the refinement"
                    # image unpadding is taken care of in the refiner, so that output image
                    # is same size as the input image
                    cur_res = refine_predict(batch, model, **predict_config.refiner)
                    cur_res = cur_res[0].permute(1,2,0).detach().cpu().numpy()
                else:
                    with torch.no_grad():
                        batch = move_to_device(batch, device)
                        batch['mask'] = (batch['mask'] > 0) * 1
                        batch = model(batch)                    
                        cur_gt = batch['image'][0].permute(1, 2, 0).detach().cpu().numpy()
                        cur_res = batch[predict_config.out_key][0].permute(1, 2, 0).detach().cpu().numpy()
                        unpad_to_size = batch.get('unpad_to_size', None)
                        if unpad_to_size is not None:
                            orig_height, orig_width = unpad_to_size
                            cur_res = cur_res[:orig_height, :orig_width]
                # assert batch size is 1 for inference 
                assert len(cur_res.shape) == 3, f"Expected 3 dimensions, got {cur_res.dim()}"
                
                
                cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
                cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
                cur_gt = np.clip(cur_gt * 255, 0, 255).astype('uint8')
                cur_gt = cv2.cvtColor(cur_gt, cv2.COLOR_RGB2BGR)
                
                # Get mask overlaid on the input image
                mask = batch['mask'][0].detach().cpu().numpy()
                mask = np.clip(mask * 255, 0, 255).astype('uint8')
                red_mask = np.zeros_like(cur_gt)
                red_mask[:, :, 2] = mask  # Assuming mask is already binary and 2D
                # In areas where there is no observation, lower the other channels by a factor (e.g., 0.5)
                overlayed_img = np.copy(cur_gt)
                overlayed_img[mask[0] > 0] = (overlayed_img[mask[0] > 0] * 0.5).astype('uint8')
                overlayed_img[:, :, 2] = np.clip(overlayed_img[:, :, 0] + 0.5 * red_mask[:, :, 2], 0, 255)
                
                # Get current input image (with unobserved masked out)
                gt_masked = np.copy(cur_gt)
                gt_masked[mask[0] > 0] = 122

            
                # display output (cur_input and cur_res stacked vertically)
                # add text to the output image (cur_res -> 'Result'), (cur_input -> 'Input')
                fontScale = 0.5
                model_name = predict_config.model.display_names[model_i]
                cur_res = cv2.putText(cur_res, 'Result {}'.format(model_name), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 0, 255), 2, cv2.LINE_AA)
                cur_input = cv2.putText(overlayed_img, 'Current Input', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 0, 255), 2, cv2.LINE_AA)
                gt_masked = cv2.putText(gt_masked, 'GT (with mask)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 0, 255), 2, cv2.LINE_AA)
                # print("cur_res.shape", cur_res.shape)
                # print("gt_masked.shape", gt_masked.shape)
                pred_result.append(cur_res)
            
            print(f"Writing to {cur_out_fname}")
            # print("cur_input.shape", cur_input.shape)
            # print("gt_masked.shape", gt_masked.shape)
            # print("pred_result[0].shape", pred_result[0].shape)
            # print("pred_result[1].shape", pred_result[1].shape)
            output = np.vstack((cur_input, gt_masked, np.vstack(pred_result)))
            # import pdb; pdb.set_trace()
            cv2.imwrite(cur_out_fname, output)

    except KeyboardInterrupt:
        LOGGER.warning('Interrupted by user')
    except Exception as ex:
        LOGGER.critical(f'Prediction failed due to {ex}:\n{traceback.format_exc()}')
        sys.exit(1)


if __name__ == '__main__':
    main()
