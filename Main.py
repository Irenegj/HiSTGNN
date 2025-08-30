import argparse
import os
import torch
import datetime
import time
import shutil
from Config import configforU_W_R, configforU_W
from Exp.Exp_U_W_R import Exp_Model as Exp_U_W_R
from Exp.Exp_U_W import Exp_Model as Exp_U_W

start = time.time()
scene = "U_W_R_HiSTGNN_5_3_2"

#A scenario with five drones, three weapon systems, and two radar system.
if scene == "U_W_R_HiSTGNN_5_3_2":
    con = configforU_W_R()
    Exp = Exp_U_W_R

#A scenario with five drones and three integrated weapon-radar systems.
elif scene == "U_W_HiSTGNN_5_3":
    con = configforU_W()
    Exp = Exp_U_W

parser = argparse.ArgumentParser(description='Assessment model')
parser.add_argument('--scene', type=str, default=scene, help='Scene tag')
parser.add_argument('--model', type=str, default=con.model, help='Model type' )
parser.add_argument('--data', type=str,  default='Dataset_database', help='Data type')
parser.add_argument('--table_name', type=str, default= con.table_name, help='The data table in the database from which the data originates')
parser.add_argument('--norm_max_min', type=bool, default=False, help='Whether to standardize')
parser.add_argument('--N_UAV', type=int, default=1, help='The number of UAV')
parser.add_argument('--N_FUAV', type=int, default=con.num_UAV-1, help='The number of friendly UAVs')
parser.add_argument('--N_SAM', type=int, default=con.num_SAM, help='The number of ground missiles')
parser.add_argument('--N_RADAR', type=int, default=con.num_RADAR, help='The number of ground missiles')
parser.add_argument('--n_en_att', type=int, default=4, help='The number of entity attributes')
parser.add_argument('--seq_len', type=int, default=20, help='Input sequence length')
parser.add_argument('--seq_len_slice', type=int, default=1, help='Path separation granularity')
parser.add_argument('--input_size', type=int, default=18, help='Input size =  Number of Receiver Attributes + Number of Sender Attributes  + Length of effect vector')
parser.add_argument('--effect_size', type=int, default=10, help='Length of effect vector')
parser.add_argument('--output_size', type=int, default=1, help='Length of output vector')
parser.add_argument('--output_length', type=int, default=1, help='Length of output sequence')
parser.add_argument('--nlayers_1', type=int, default=1, help='The number of layers of the GRU layer in relation module')
parser.add_argument('--nlayers_2', type=int, default=1, help='The number of layers of the GRU layer in evaluation module')
parser.add_argument('--hidden_size_1', type=int, default=64, help='The number of hidden nodes in the relation')
parser.add_argument('--hidden_size_2', type=int, default=64,  help='The number of hidden nodes in the evaluation')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate')
parser.add_argument('--loss', type=str, default='mse',help='Loss function')
parser.add_argument('--clip', type=float, default=0.01,  help='Gradient clipping')
parser.add_argument('--epochs', type=int, default=200, help='Upper epoch limit')
parser.add_argument('--itr', type=int, default=5, help='Experiments times')
parser.add_argument('--batch_size', type=int, default =128, metavar='N', help='Batch Size')
parser.add_argument('--eval_batch_size', type=int, default=128, metavar='N', help='Test batch size')
parser.add_argument('--test_batch_size', type=int, default=1, metavar='N',  help='Eval batch size')
parser.add_argument('--n_tra', type=int, default=409600, help='The number of trajectories (Total amount of data used for training)')
parser.add_argument('--n_vali', type=int, default=89600,  help='The number of trajectories (Total amount of data used for validation)')
parser.add_argument('--n_test', type=int, default=5, help='The number of trajectories (Total amount of data used for testing)')
parser.add_argument('--lradj', type=str, default='type1',help='Adjust learning rate')
parser.add_argument('--num_workers', type=int, default=0, help='Data loader num workers')
parser.add_argument('--seed', type=int, default=None, help='Random seed')
parser.add_argument('--log-interval', type=int, default=50, metavar='N', help='report interval')
parser.add_argument('--save', type=str, default='model.pt', help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='', help='path to export the final model in onnx format')
parser.add_argument('--name', type=str, default=None, help='The name specified for this experiment will be used to generate a folder.')
parser.add_argument('--output', type=str, default='./output', help='Output folder')
parser.add_argument('--use_gpu', type=bool, default=False, help='Use gpu')
parser.add_argument('--gpu', type=int, default=0, help='Gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='Use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3',help='Device ids of multile gpus')
args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False



if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    print("GPU Device ID：", args.device_ids)
    args.gpu = args.device_ids[0]

now_time = datetime.datetime.now().strftime('%mM_%dD %HH:%Mm:%Ss').replace(" ", "_").replace(":", "_")

# Create Storage Path
run_name_dir_old = args.model +"_"+args.data+ "_" + str(args.itr) + "_" + now_time
args.output = os.path.join(args.output, args.scene)
run_name_dir = os.path.join(args.output, run_name_dir_old)
if not os.path.exists(run_name_dir):
    os.makedirs(run_name_dir)
df_columns = []
test_columns = []

for ii in range(args.itr):
    print("-------------Experiment #{}------------".format(ii+1))
    run_ex_dir = os.path.join(run_name_dir, "_{}_th experiment record".format(ii + 1))
    if not os.path.exists(run_ex_dir):
        os.makedirs(run_ex_dir)
    for file in os.listdir(os.getcwd()):
        if file.endswith("n.py") or  file.endswith("g.py"):
            shutil.copy2(file, run_ex_dir)
    for file in os.listdir(os.path.join(os.getcwd(),'Framework')):
        if file.endswith(".py"):
            shutil.copy2(os.path.join(os.path.join(os.getcwd(),'Framework'),file), run_ex_dir)

    logger_train = open(os.path.join(run_ex_dir , 'train_log.txt'), 'w+', encoding='UTF-8')
    logger_test = open(os.path.join(run_ex_dir , 'test_log.txt'), 'w+', encoding='UTF-8')
    # save args to logger
    logger_train.write(str(args) + '\n')
    exp = Exp(args,con)
    print('>>>>>>>start training :  {}  >>>>>>>>>>>>>>>>>>>>>>>>>>'.format(run_name_dir_old))
    model, all_epoch_train_loss, all_epoch_vali_loss, all_epoch_test_loss, epoch_count = exp.train(
        logger_train, run_ex_dir)
    print('>>>>>>>testing :  {}  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(run_name_dir_old))
    test_pred, test_true = exp.test(logger_test, run_ex_dir)
    torch.cuda.empty_cache()
end = time.time()
print("execution time：\t",str(end-start))