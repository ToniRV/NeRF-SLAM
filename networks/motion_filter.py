import torch
import lietorch

from .geom import projective_ops as pops
from .modules.corr import CorrBlock


# Basically populates the video with the frames that have enough motion
# It calculates the features for each frame, does one step GRU update
# to get a sense of the flow and if the flow > min_flow
# it adds the frame, together with features and context to the video object
class MotionFilter:
    """ This class is used to filter incoming frames and extract features """
    def __init__(self, net, video, min_flow_thresh=2.5, device="cuda:0"):
        # split net modules
        self.context_net = net.cnet
        self.feature_net = net.fnet
        self.update_net = net.update

        self.video = video
        self.min_flow_thresh = min_flow_thresh

        self.skipped_frames = 0
        self.device = device

        # mean, std for image normalization
        self.MEAN = torch.as_tensor([0.485, 0.456, 0.406], device=self.device)[:, None, None]
        self.STDV = torch.as_tensor([0.229, 0.224, 0.225], device=self.device)[:, None, None]
        
    @torch.cuda.amp.autocast(enabled=True)
    def __context_encoder(self, image):
        """ context features """
        context_maps, gru_input_maps = self.context_net(image).split([128,128], dim=2)
        return context_maps.tanh().squeeze(0), gru_input_maps.relu().squeeze(0)

    @torch.cuda.amp.autocast(enabled=True)
    def __feature_encoder(self, image):
        """ features for correlation volume """
        return self.feature_net(image).squeeze(0)

    @torch.cuda.amp.autocast(enabled=True)
    @torch.no_grad()
    def track(self, k, timestamp, image, depth=None, intrinsics=None):
        """ main update operation - run on every frame in video """

        # normalize images
        img_normalized = image[None, :, [2,1,0]].to(self.device) / 255.0
        img_normalized = img_normalized.sub_(self.MEAN).div_(self.STDV)

        # extract features
        feature_map = self.__feature_encoder(img_normalized)

        ### always add first frame to the depth video ###
        if k == 0:
            self.add_frame_to_video(timestamp, image, img_normalized, feature_map, depth, intrinsics)
        ### only add new frame if there is enough motion ###
        else:                
            ht = image.shape[-2] // 8
            wd = image.shape[-1] // 8

            # Index correlation volume
            coords0 = pops.coords_grid(ht, wd, device=self.device)[None,None]
            corr = CorrBlock(self.feature_maps[None,[0]], feature_map[None,[0]])(coords0) # TODO why not send the corr block?

            # Approximate flow magnitude using 1 update iteration
            _, delta, weight = self.update_net(self.context_maps[None], self.gru_input_maps[None], corr)

            # Check motion magnitue / add new frame to video
            has_enough_motion = delta.norm(dim=-1).mean().item() > self.min_flow_thresh
            if has_enough_motion:
                self.add_frame_to_video(timestamp, image, img_normalized, feature_map, depth, intrinsics)
                self.skipped_frames = 0
            else:
                self.skipped_frames += 1
        
    # Save the network responses (feature + context) and image for later inference
    def add_frame_to_video(self, timestamp, image, img_normalized, feature_map, depth_img=None, intrinsics=None):
        context_maps, gru_input_maps = self.__context_encoder(img_normalized[:,[0]])
        self.context_maps, self.gru_input_maps, self.feature_maps = context_maps, gru_input_maps, feature_map
        identity_pose = lietorch.SE3.Identity(1,).data.squeeze()
        self.video.append(timestamp, image[0], identity_pose,
                            1.0, # Initialize disps at 1
                            depth_img, # If available
                            intrinsics / 8.0,
                            feature_map, context_maps[0,0], gru_input_maps[0,0])
