import os
import logging



#######################  analysis　　############################################################
channel_derain = 64
################################################################################################

#######################  analysis　############################################################
sk = True
#####################################################################
block_style = 'msrb'  # sk, concat_scale
pyramid = False
scale_num = 3
msdc = True
gan = False
################################################################3
dense = True


##############Concat#########################

#################################33
guided_in_block = True

concat_num = 2
#################################################################################################

##############################  analysis---1  ################################
network_style = 'rain_derain_with_guide'
# only_img
# only_rain,
# only_derain,
# rain_derain_no_guide,
# rain_derain_with_guide,
# rain_derain_with_guide



# rain_derain_with_inner_guide_and_dis
l_rain = 0.5
l_derain = 0.5
l_rain_derain = 0.01
l_dis_rain_derain = 0.1
l_dis_img = 0.25


########################################################################################

aug_data = False # Set as False for fair comparison

patch_size = 160
sizePatchGAN = int(patch_size/32)
lr = 5e-4

data_dir = '/mnt/lustre/zhangyajie/workspace/cvpr/data/rain100H'
log_dir = '../logdir'
show_dir = '../showdir'
model_dir = '../models'
show_dir_feature = '../showdir_feature'

log_level = 'info'
model_path = os.path.join(model_dir, 'latest')
save_steps = 400

num_workers = 16
num_GPU = 8
device_id = ''
for i in range(num_GPU):
    device_id += str(i) + ','

root_dir = os.path.join(data_dir, 'train')
mat_files = os.listdir(root_dir)
num_datasets = len(mat_files)

epoch = 2000
batch_size = 32

l1 = int(3/5 * epoch * num_datasets / batch_size)
l2 = int(4/5 * epoch * num_datasets / batch_size)
total_step = int((epoch * num_datasets)/batch_size)
one_epoch_step = int(num_datasets/batch_size)


logger = logging.getLogger('train')
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


