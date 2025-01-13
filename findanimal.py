import sys
sys.path.append("..")
sys.path.append("./sam")
from sam.segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import numpy as np
from tool.segmentor import Segmentor
from tool.detector import Detector
from tool.transfer_tools import draw_outline, draw_points
import cv2
from scipy.ndimage import binary_dilation
from PIL import Image
import argparse

sam_args = {    
    'sam_checkpoint': "ckpt/sam_vit_h_4b8939.pth",
    'model_type': "vit_h",
    'generator_args':{
        'points_per_side': 32,
        'pred_iou_thresh': 0.8,
        'stability_score_thresh': 0.9,
        'crop_n_layers': 1,
        'crop_n_points_downscale_factor': 2,
        'min_mask_region_area': 200,
    },
    'gpu_id': 0,
}
tracker_args = {
    'sam_gap': 10, # the interval to run sam to segment new objects
    'min_area': 200, # minimal mask area to add a new mask as a new object
    'max_obj_num': 255, # maximal object number to track in a video
    'min_new_obj_iou': 0.8, # the background area ratio of a new object should > 80% 
}

np.random.seed(200)
_palette = ((np.random.random((3*255))*0.7+0.3)*255).astype(np.uint8).tolist()
_palette = [0,0,0]+_palette

def colorize_mask(pred_mask):
    save_mask = Image.fromarray(pred_mask.astype(np.uint8))
    save_mask = save_mask.convert(mode='P')
    save_mask.putpalette(_palette)
    save_mask = save_mask.convert(mode='RGB')
    return np.array(save_mask)

def draw_mask(img, mask, alpha=0.5, id_countour=False):
    img_mask = np.zeros_like(img)
    img_mask = img
    if id_countour:
        # very slow ~ 1s per image
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[obj_ids!=0]

        for id in obj_ids:
            # Overlay color on  binary mask
            if id <= 255:
                color = _palette[id*3:id*3+3]
            else:
                color = [0,0,0]
            foreground = img * (1-alpha) + np.ones_like(img) * alpha * np.array(color)
            binary_mask = (mask == id)

            # Compose image
            img_mask[binary_mask] = foreground[binary_mask]

            countours = binary_dilation(binary_mask,iterations=1) ^ binary_mask
            img_mask[countours, :] = 0
    else:
        binary_mask = (mask!=0)
        countours = binary_dilation(binary_mask,iterations=1) ^ binary_mask
        foreground = img*(1-alpha)+colorize_mask(mask)*alpha
        img_mask[binary_mask] = foreground[binary_mask]
        img_mask[countours,:] = 0
        
    return img_mask.astype(img.dtype)
class SegSearch():
    def __init__(self,segtracker_args, sam_args) -> None:
        """
         Initialize SAM and AOT.
        """
        self.sam = Segmentor(sam_args)
        self.detector = Detector(self.sam.device)
        self.sam_gap = segtracker_args['sam_gap']
        self.min_area = segtracker_args['min_area']
        self.max_obj_num = segtracker_args['max_obj_num']
        self.min_new_obj_iou = segtracker_args['min_new_obj_iou']
        self.reference_objs_list = []
        self.object_idx = 1
        self.curr_idx = 1
        self.origin_merged_mask = None  # init by segment-everything or update
        self.first_frame_mask = None

        # debug
        self.everything_points = []
        self.everything_labels = []
        print("SegTracker has been initialized")

    def seg(self,frame):
        '''
        Arguments:
            frame: numpy array (h,w,3)
        Return:
            origin_merged_mask: numpy array (h,w)
        '''
        frame = frame[:, :, ::-1]
        anns = self.sam.everything_generator.generate(frame)

        # anns is a list recording all predictions in an image
        if len(anns) == 0:
            return
        # merge all predictions into one mask (h,w)
        # note that the merged mask may lost some objects due to the overlapping
        self.origin_merged_mask = np.zeros(anns[0]['segmentation'].shape,dtype=np.uint8)
        idx = 1
        for ann in anns:
            if ann['area'] > self.min_area:
                m = ann['segmentation']
                self.origin_merged_mask[m==1] = idx
                idx += 1
                self.everything_points.append(ann["point_coords"][0])
                self.everything_labels.append(1)

        obj_ids = np.unique(self.origin_merged_mask)
        obj_ids = obj_ids[obj_ids!=0]

        self.object_idx = 1
        for id in obj_ids:
            if np.sum(self.origin_merged_mask==id) < self.min_area or self.object_idx > self.max_obj_num:
                self.origin_merged_mask[self.origin_merged_mask==id] = 0
            else:
                self.origin_merged_mask[self.origin_merged_mask==id] = self.object_idx
                self.object_idx += 1

        self.first_frame_mask = self.origin_merged_mask
        return self.origin_merged_mask

    def update_origin_merged_mask(self, updated_merged_mask):
        self.origin_merged_mask = updated_merged_mask
        # obj_ids = np.unique(updated_merged_mask)
        # obj_ids = obj_ids[obj_ids!=0]
        # self.object_idx = int(max(obj_ids)) + 1

    def reset_origin_merged_mask(self, mask, id):
        self.origin_merged_mask = mask
        self.curr_idx = id

    def add_reference(self,frame,mask,frame_step=0):
        '''
        Add objects in a mask for tracking.
        Arguments:
            frame: numpy array (h,w,3)
            mask: numpy array (h,w)
        '''
        self.reference_objs_list.append(np.unique(mask))
        self.curr_idx = self.get_obj_num()
    
    def get_tracking_objs(self):
        objs = set()
        for ref in self.reference_objs_list:
            objs.update(set(ref))
        objs = list(sorted(list(objs)))
        objs = [i for i in objs if i!=0]
        return objs
    
    def get_obj_num(self):
        objs = self.get_tracking_objs()
        if len(objs) == 0: return 0
        return int(max(objs))

    def find_new_objs(self, track_mask, seg_mask):
        '''
        Compare tracked results from AOT with segmented results from SAM. Select objects from background if they are not tracked.
        Arguments:
            track_mask: numpy array (h,w)
            seg_mask: numpy array (h,w)
        Return:
            new_obj_mask: numpy array (h,w)
        '''
        new_obj_mask = (track_mask==0) * seg_mask
        new_obj_ids = np.unique(new_obj_mask)
        new_obj_ids = new_obj_ids[new_obj_ids!=0]
        # obj_num = self.get_obj_num() + 1
        obj_num = self.curr_idx
        for idx in new_obj_ids:
            new_obj_area = np.sum(new_obj_mask==idx)
            obj_area = np.sum(seg_mask==idx)
            if new_obj_area/obj_area < self.min_new_obj_iou or new_obj_area < self.min_area\
                or obj_num > self.max_obj_num:
                new_obj_mask[new_obj_mask==idx] = 0
            else:
                new_obj_mask[new_obj_mask==idx] = obj_num
                obj_num += 1
        return new_obj_mask
        

    def seg_acc_bbox(self, origin_frame: np.ndarray, bbox: np.ndarray,):
        ''''
        Use bbox-prompt to get mask
        Parameters:
            origin_frame: H, W, C
            bbox: [[x0, y0], [x1, y1]]
        Return:
            refined_merged_mask: numpy array (h, w)
            masked_frame: numpy array (h, w, c)
        '''
        # get interactive_mask
        interactive_mask = self.sam.segment_with_box(origin_frame, bbox)[0]
        refined_merged_mask = self.add_mask(interactive_mask)

        # draw mask
        masked_frame = draw_mask(origin_frame.copy(), refined_merged_mask)

        # draw bbox
        masked_frame = cv2.rectangle(masked_frame, bbox[0], bbox[1], (0, 0, 255))

        return refined_merged_mask, masked_frame

    def seg_acc_click(self, origin_frame: np.ndarray, coords: np.ndarray, modes: np.ndarray, multimask=True):
        '''
        Use point-prompt to get mask
        Parameters:
            origin_frame: H, W, C
            coords: nd.array [[x, y]]
            modes: nd.array [[1]]
        Return:
            refined_merged_mask: numpy array (h, w)
            masked_frame: numpy array (h, w, c)
        '''
        # get interactive_mask
        interactive_mask = self.sam.segment_with_click(origin_frame, coords, modes, multimask)

        refined_merged_mask = self.add_mask(interactive_mask)

        # draw mask
        masked_frame = draw_mask(origin_frame.copy(), refined_merged_mask)

        # draw points
        # self.everything_labels = np.array(self.everything_labels).astype(np.int64)
        # self.everything_points = np.array(self.everything_points).astype(np.int64)

        masked_frame = draw_points(coords, modes, masked_frame)

        # draw outline
        masked_frame = draw_outline(interactive_mask, masked_frame)

        return refined_merged_mask, masked_frame

    def add_mask(self, interactive_mask: np.ndarray):
        '''
        Merge interactive mask with self.origin_merged_mask
        Parameters:
            interactive_mask: numpy array (h, w)
        Return:
            refined_merged_mask: numpy array (h, w)
        '''
        if self.origin_merged_mask is None:
            self.origin_merged_mask = np.zeros(interactive_mask.shape,dtype=np.uint8)

        refined_merged_mask = self.origin_merged_mask.copy()
        refined_merged_mask[interactive_mask > 0] = self.curr_idx

        return refined_merged_mask
    
    def search_and_seg(self, origin_frame: np.ndarray, grounding_caption, box_threshold, text_threshold, box_size_threshold=1, reset_image=False):
        '''
        Using Grounding-DINO to detect object acc Text-prompts
        Retrun:
            refined_merged_mask: numpy array (h, w)
            annotated_frame: numpy array (h, w, 3)
        '''
        # backup id and origin-merged-mask
        bc_id = self.curr_idx
        bc_mask = self.origin_merged_mask

        # get annotated_frame and boxes
        annotated_frame, boxes = self.detector.run_grounding(origin_frame, grounding_caption, box_threshold, text_threshold)
        
        for i in range(len(boxes)):
            bbox = boxes[i]
            if (bbox[1][0] - bbox[0][0]) * (bbox[1][1] - bbox[0][1]) > annotated_frame.shape[0] * annotated_frame.shape[1] * box_size_threshold:
                continue
            interactive_mask = self.sam.segment_with_box(origin_frame, bbox, reset_image)[0]
            refined_merged_mask = self.add_mask(interactive_mask)
            self.update_origin_merged_mask(refined_merged_mask)
            self.curr_idx += 1
            image = origin_frame

            originmask = np.zeros(origin_frame.shape[:2],dtype=np.uint8)
            originmask[interactive_mask > 0] = 255#i+1
            masked = cv2.bitwise_and(image, image, mask=originmask)

            masked = cv2.cvtColor(masked, cv2.COLOR_RGB2BGR)
            cv2.imwrite('./result/mask_'+ f'{i:02d}.png', masked)

        # reset origin_mask
        self.reset_origin_merged_mask(bc_mask, bc_id)

        return refined_merged_mask, annotated_frame

if __name__ == '__main__':    

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="./test03.jpeg", help="source file path for finding animal")
    args = parser.parse_args()

    SearchAnimal = SegSearch(tracker_args, sam_args)
    
    # ------------------ search and segmentation test ----------------------
    
    orgframe = cv2.imread(args.source)
    orgframe = cv2.cvtColor(orgframe, cv2.COLOR_BGR2RGB)
    txtprompt = "animal"
    boxthreshold = 0.35
    textthreshold = 0.25

    predicted, annotated = SearchAnimal.search_and_seg(orgframe, txtprompt, boxthreshold, textthreshold)
    
    annotated = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
    masked = draw_mask(annotated, predicted)    
    cv2.imwrite('./result/masked_frame.png', masked)


    
    