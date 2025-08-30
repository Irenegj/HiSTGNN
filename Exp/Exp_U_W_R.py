
import os
import time
from utils.metrics import metric
import numpy as np
import torch
import torch.nn as nn
from Exp.Exp_basic import Exp_Basic
from Data.Data_loader import Dataset_database_U_W_R as Dataset_database
from torch.utils.data import DataLoader
from Framework.HiSTGNN_U_W_R import Model_layer as HiSTGNN_U_W_R

class Exp_Model(Exp_Basic):
    def __init__(self, args,con):
        super(Exp_Model, self).__init__(args)
        self.con = con
        self.rs_12, self.rr_12, self.rs_23, self.rr_23, self.rs_34, self.rr_34, self.r_type_12, self.r_type_23, self.r_type_34 = self._get_graph_paras()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        model_dict = {
            'HiSTGNN_U_W_R': HiSTGNN_U_W_R,
        }
        model = model_dict[self.args.model](
            self.args.input_size,
            self.args.effect_size,
            self.args.output_size,
            self.args.output_length,
            self.args.nlayers_1,
            self.args.nlayers_2,
            self.args.hidden_size_1,
            self.args.hidden_size_2,
        ).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def CFtest_msg_creat(self, run_name_dir_old):
        T = ["CF-" + run_name_dir_old + "\n"]
        for attr, value in vars(self.con).items():
            T.append(attr + ": " + str(value) + "\n")
        return T

    def get_r(self, objects, relations):
        n_object = len(objects)
        n_relation = len(relations)
        r = torch.zeros(n_relation, n_object)
        for index, i in enumerate(relations):
            r[index][i-1] = 1
        return r

    def count_values(self, dictionary, f):
        count = 0
        for value in dictionary.values():
            if f in value:
                count += 1
        return count

    def get_r_type(self, relations_type):
        r_type = []
        for key, value in relations_type.items():
            r_type.append([value["type"],value["num"]])
        return r_type

    def _get_graph_paras (self):
        rs_12 = self.get_r(self.con.objects_1, self.con.relations_s_12)
        rs_23 = self.get_r(self.con.objects_2, self.con.relations_s_23)
        rs_34 = self.get_r(self.con.objects_3, self.con.relations_s_34)
        rr_12 = self.get_r(self.con.objects_2, self.con.relations_r_12)
        rr_23 = self.get_r(self.con.objects_3, self.con.relations_r_23)
        rr_34 = self.get_r(self.con.objects_4, self.con.relations_r_34)
        r_type_12 = self.get_r_type(self.con.relations_type_12)
        r_type_23 = self.get_r_type(self.con.relations_type_23)
        r_type_34 = self.get_r_type(self.con.relations_type_34)
        return rs_12, rr_12, rs_23, rr_23, rs_34, rr_34, r_type_12, r_type_23, r_type_34

    def _dim_expansion(self,x,batch_size):
        x = x.unsqueeze(1).unsqueeze(1).repeat(1,self.args.seq_len,batch_size,1)
        return x

    def get_num(self,n_tra, n_vali,n_test, num):
        point_one =  int(n_tra/num) + 1
        point_two = point_one + int(n_vali/num) +1
        point_three = point_two + int(n_test/num) + 1
        return  point_one,  point_two , point_three

    def get_numfortest(self,n_test,num):
        point_one =  int(n_test/num) + 1
        return  point_one

    def _get_data(self, flag):
        con = self.con
        args = self.args
        data_dict = {
            'Dataset_database': Dataset_database
        }
        Data = data_dict[self.args.data]
        point_one, point_two, point_three = self.get_num(args.n_tra, args.n_vali, args.n_test, (args.N_UAV+args.N_FUAV))
        if flag == 'test':
            shuffle_flag = False;
            drop_last = True;
            start_n = point_two
            end_n = point_three
            n = args.n_test
            batch_size = args.test_batch_size;
        elif flag == 'val':
            shuffle_flag = False;
            drop_last = True;
            start_n = point_one
            end_n = point_two
            n = args.n_vali
            batch_size = args.eval_batch_size;
        else:
            shuffle_flag = True;
            drop_last = True;
            start_n = 0
            n = args.n_tra
            end_n = point_one
            batch_size = args.batch_size;

        data_set = Data(
            n_UAV = args.N_UAV + args.N_FUAV,
            n_SAM = args.N_SAM,
            n_RADAR = args.N_RADAR,
            table_name = args.table_name,
            start_n = start_n,
            end_n = end_n,
            n = n,
            con = con,
            seq_len=args.seq_len,
            length_slice=args.seq_len_slice,
            n_en_att=args.n_en_att,
            output_length=args.output_length,
            norm_max_min=args.norm_max_min
        )
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader

    def _get_datafortest(self, flag):
        con = self.con
        args = self.args
        data_dict = {
            'Dataset_database': Dataset_database
        }
        Data = data_dict[self.args.data]
        point_one = self.get_numfortest(args.n_test, (args.N_UAV + args.N_FUAV))

        if flag == 'test':
            shuffle_flag = False;
            drop_last = True;
            start_n = 0
            end_n = point_one
            n = args.n_test
            batch_size = args.test_batch_size;

        data_set = Data(
            n_UAV=args.N_UAV + args.N_FUAV,
            n_SAM=args.N_SAM,
            n_RADAR=args.N_RADAR,
            table_name=args.table_name,
            start_n=start_n,
            end_n=end_n,
            n=n,
            con=con,
            seq_len=args.seq_len,
            length_slice=args.seq_len_slice,
            n_en_att=args.n_en_att,
            output_length=args.output_length,
            norm_max_min=args.norm_max_min
        )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        if self.args.loss == 'mse':
            criterion = nn.MSELoss()
        if self.args.loss == 'L1loss':
            criterion = nn.L1Loss()
        if self.args.loss == 'huberloss':
            criterion = nn.SmoothL1Loss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        batch_size = 1
        (rs_12, rr_12, rs_23, rr_23,  rs_34, rr_34) = (self._dim_expansion(x, batch_size) for x in
                                        (self.rs_12, self.rr_12, self.rs_23, self.rr_23, self.rs_34, self.rr_34))
        self.model.eval()
        total_loss = []
        for i, (batch_en_att_1,batch_en_att_2,batch_en_att_3,batch_en_att_4, batch_label) in enumerate(vali_loader):
            pred = self.model(batch_en_att_1, batch_en_att_2, batch_en_att_3, batch_en_att_4, rs_12, rr_12,
                              rs_23, rr_23, rs_34, rr_34, self.r_type_12, self.r_type_23, self.r_type_34)
            pred  = torch.permute(pred, (1, 0, 2))
            loss = criterion(pred.detach().cpu(), batch_label.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def tes(self, tes_data, tes_loader, criterion, logger_train):
        batch_size = 1
        (rs_12, rr_12, rs_23, rr_23,  rs_34, rr_34) = (self._dim_expansion(x, batch_size) for x in
                                        (self.rs_12, self.rr_12, self.rs_23, self.rr_23, self.rs_34, self.rr_34))
        self.model.eval()
        total_loss = []
        for i, (batch_en_att_1,batch_en_att_2,batch_en_att_3,batch_en_att_4, batch_label) in enumerate(tes_loader):
            pred = self.model(batch_en_att_1, batch_en_att_2, batch_en_att_3, batch_en_att_4, rs_12, rr_12,
                              rs_23, rr_23, rs_34, rr_34, self.r_type_12, self.r_type_23, self.r_type_34)
            pred  = torch.permute(pred, (1, 0, 2))
            logger_train.flush()
            loss = criterion(pred.detach().cpu(), batch_label.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, logger_train, run_ex_dir):
        batch_size = self.args.batch_size
        (rs_12, rr_12, rs_23, rr_23, rs_34, rr_34) = (self._dim_expansion(x, batch_size) for x in
                                                      (self.rs_12, self.rr_12, self.rs_23, self.rr_23, self.rs_34,
                                                       self.rr_34))
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        time_now = time.time()
        train_steps = len(train_loader)
        model_optim = self._select_optimizer()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optim, 'min', factor=0.5, patience=self.con.patience)
        criterion = self._select_criterion()
        all_epoch_train_loss = []
        all_epoch_vali_loss = []
        all_epoch_test_loss = []
        epoch_count = 0
        vali_loss_min = 100
        for epoch in range(self.args.epochs):
            epoch_count += 1
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, (batch_en_att_1,batch_en_att_2,batch_en_att_3,batch_en_att_4, batch_label) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                pred = self.model(batch_en_att_1, batch_en_att_2, batch_en_att_3, batch_en_att_4, rs_12, rr_12,
                                  rs_23, rr_23, rs_34, rr_34, self.r_type_12, self.r_type_23, self.r_type_34)
                pred = torch.permute(pred, (1, 0, 2))
                loss = criterion(pred.float(), batch_label.float())
                train_loss.append(loss.item())
                if (iter_count  + 1) % 100 == 0:
                    printlog = "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item())
                    print(printlog)
                    logger_train.write(printlog + '\n')
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.epochs - epoch) * train_steps - i)
                    printlog = '\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time)
                    print(printlog)
                    logger_train.write(printlog + '\n')
                    logger_train.flush()
                    iter_count = 0
                    time_now = time.time()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
                model_optim.step()
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            scheduler.step(vali_loss)
            test_loss = self.tes(test_data, test_loader, criterion,logger_train)
            all_epoch_train_loss.append(float(round(train_loss, 1)))
            all_epoch_vali_loss.append(float(round(vali_loss, 1)))
            all_epoch_test_loss.append(float(round(test_loss, 1)))
            printlog = "Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time)
            print(printlog)

            logger_train.write(printlog + '\n')
            current_lr = model_optim.param_groups[0]['lr']
            printlog = "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f} Lr: {5:f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss, current_lr)
            print(printlog)
            logger_train.write(printlog + '\n')
            logger_train.flush()
            if vali_loss < vali_loss_min:
                vali_loss_min = vali_loss
                with open(os.path.join(run_ex_dir, 'model.pt'), 'wb') as f:
                    torch.save(self.model, f)
            if current_lr < 0.00001:
                break
        printlog = '【训练】本次实验训练的train平均损失:%f'%round(float(np.mean(all_epoch_train_loss)), 1)
        print(printlog)
        logger_train.write(printlog + '\n')
        printlog = '【验证】本次实验训练的vali平均损失:%f' % round(float(np.mean(all_epoch_vali_loss)), 1)
        print(printlog)
        logger_train.write(printlog + '\n')
        printlog = '【验证】本次实验训练的test平均损失:%f' % round(float(np.mean(all_epoch_test_loss)), 1)
        print(printlog)
        logger_train.write(printlog + '\n')
        logger_train.write("----实际训练的epoch:%f----" % epoch_count)
        logger_train.flush()
        return self.model,all_epoch_train_loss, all_epoch_vali_loss, all_epoch_test_loss, epoch_count

    def test(self, logger_test,run_ex_dir):
        test_data, test_loader = self._get_data(flag='test')
        batch_size = 1
        (rs_12, rr_12, rs_23, rr_23, rs_34, rr_34) = (self._dim_expansion(x, batch_size) for x in
                                                      (self.rs_12, self.rr_12, self.rs_23, self.rr_23, self.rs_34,
                                                       self.rr_34))
        self.model.eval()
        preds = []
        trues = []
        for i, (batch_en_att_1,batch_en_att_2,batch_en_att_3,batch_en_att_4,batch_label) in enumerate(test_loader):
            pred = self.model(batch_en_att_1, batch_en_att_2, batch_en_att_3, batch_en_att_4, rs_12, rr_12,
                              rs_23, rr_23, rs_34, rr_34, self.r_type_12, self.r_type_23, self.r_type_34)
            pred = torch.permute(pred, (1, 0, 2))
            preds.append(pred.detach().cpu().numpy())
            trues.append(batch_label.detach().cpu().numpy())
        preds = np.array(preds)
        trues = np.array(trues)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        folder_path = run_ex_dir + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        mae,mse,rmse = metric(preds, trues)
        mae,mse,rmse = round(float(mae), 1), round(float(mse), 1), round(float(rmse), 1)
        print(
            '测试集评估结果：\t平均绝对误差 MAE:{}，均方误差MSE:{}，均方根误差RMSE:{} \n'.format(mae,mse,rmse))
        logger_test.write("【评估】本次实验的test集平均绝对误差MAE:%f" % mae+ '\n')
        logger_test.write("【评估】本次实验的test集均方误差MAE:%f" % mse  + '\n')
        logger_test.write("【评估】本次实验的test集均方根误差RMSE:%f" % rmse + '\n')
        logger_test.writelines(self.CFtest_msg_creat(run_ex_dir))
        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        return preds, trues







