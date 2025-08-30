from torch.utils.data import Dataset
import torch
from Data.Engine import engine


class Dataset_database_U_W(Dataset):
    def __init__(self,n_UAV, n_SAM, n_RADAR, table_name, con, seq_len, n_en_att, start_n, end_n, n, output_length, norm_max_min):
        self.n_UAV = n_UAV
        self.n_SAM = n_SAM
        self.n_RADAR = n_RADAR
        self.table_name = table_name
        self.con = con
        self.objects_1 = con.objects_1
        self.objects_2 = con.objects_2
        self.objects_3 = con.objects_3
        self.seq_len = seq_len
        self.n_en_att = n_en_att
        self.start_n = start_n
        self.end_n = end_n
        self.n = n
        self.output_length = output_length
        self.norm_max_min = norm_max_min
        self.__read_data__()

    def data_integra(self,UAVs, SAMs, P_lives):
        UAV = UAVs[:,0]
        FUAVs = UAVs[:,1:]
        SAMs_new = SAMs
        P = P_lives[:,0]
        for i in range(1,self.n_UAV):
            UAV = torch.cat((UAV,UAVs[:,i]),dim=0)
            FUAVs = torch.cat((FUAVs, UAVs[:, ~(torch.arange(UAVs.size(1)) == i).to(torch.bool)]), dim=0)
            SAMs_new  = torch.cat((SAMs_new ,SAMs),dim=0)
            P = torch.cat((P,P_lives[:,i]),dim = 0)
        return UAV[:self.n],FUAVs[:self.n],SAMs_new[:self.n],P[:self.n]

    def get_entities(self,UAV, FUAVs, SAMs, objects):
        for key, value in objects.items():
            tag = value[0]
            num = int(value[-1])
            if key == 1:
                if tag == "U":en_att = UAV.unsqueeze(1)
                elif tag == "F":en_att = FUAVs[:,num-1].unsqueeze(1)
                else: en_att = SAMs[:,num-1].unsqueeze(1)
            else:
                if tag == "U": en_att = torch.cat((en_att, UAV.unsqueeze(1)),dim = 1)
                elif tag == "F": en_att = torch.cat((en_att, FUAVs[:,num - 1].unsqueeze(1)),dim = 1)
                else: en_att = torch.cat((en_att, SAMs[:,num - 1].unsqueeze(1)),dim = 1)
        return en_att

    def get_label(self,P):
        num = len(P[0])
        label = P[:,num - self.output_length:num].unsqueeze(-1)
        return label

    def __read_data__(self):
        en = engine(self.n_UAV, self.n_SAM, self.n_RADAR, self.table_name, self.start_n, self.end_n, self.seq_len, self.n_en_att)
        UAVs, SAMs, _, P_lives = en.get_data(self.norm_max_min, self.con)
        UAV, FUAV, SAMS, P = self.data_integra(torch.tensor(UAVs), torch.tensor(SAMs), torch.tensor(P_lives))
        self.en_att_1 = self.get_entities(UAV, FUAV, SAMS, self.objects_1)
        self.en_att_2 = self.get_entities(UAV, FUAV, SAMS, self.objects_2)
        self.en_att_3 = self.get_entities(UAV, FUAV, SAMS, self.objects_3)
        self.label = self.get_label(P)*10

    def __getitem__(self, index):
        '''
        :param en_att_1: [n_entities_1l,seq_len,n_en_att]
        :param en_att_2: [n_entities_2l,seq_len,n_en_att]
        :param en_att_3: [n_entities_3l,seq_len,n_en_att]
        '''
        en_att_1_one = self.en_att_1[index]
        en_att_2_one = self.en_att_2[index]
        en_att_3_one = self.en_att_3[index]
        label_one = self.label[index]

        return en_att_1_one,en_att_2_one,en_att_3_one,label_one

    def __len__(self):
        return len(self.label)
    
class Dataset_database_U_W_R(Dataset):
    def __init__(self, n_UAV, n_SAM, n_RADAR, table_name, con, seq_len,  length_slice, n_en_att, start_n, end_n, n, output_length,
                 norm_max_min):
        self.n_UAV = n_UAV
        self.n_SAM = n_SAM
        self.n_RADAR = n_RADAR
        self.table_name = table_name
        self.con = con
        self.objects_1 = con.objects_1
        self.objects_2 = con.objects_2
        self.objects_3 = con.objects_3
        self.objects_4 = con.objects_4
        self.len_slice = length_slice
        if self.len_slice:self.seq_len = seq_len*length_slice
        else: self.seq_len = seq_len

        self.n_en_att = n_en_att
        self.start_n = start_n
        self.end_n = end_n
        self.n = n
        self.output_length = output_length
        self.norm_max_min = norm_max_min
        self.__read_data__()

    def data_integra(self, UAVs, SAMs, RADARs, P_lives):
        UAV = UAVs[:, 0]
        FUAVs = UAVs[:, 1:]
        SAMs_new = SAMs
        RADARs_new = RADARs
        P = P_lives[:, 0]
        for i in range(1, self.n_UAV):
            UAV = torch.cat((UAV, UAVs[:, i]), dim=0)
            FUAVs = torch.cat((FUAVs, UAVs[:, ~(torch.arange(UAVs.size(1)) == i).to(torch.bool)]), dim=0)
            SAMs_new = torch.cat((SAMs_new, SAMs), dim=0)
            RADARs_new = torch.cat((RADARs_new, RADARs), dim=0)
            P = torch.cat((P, P_lives[:, i]), dim=0)
        return UAV[:self.n], FUAVs[:self.n], SAMs_new[:self.n], RADARs_new[:self.n], P[:self.n]

    def get_entities(self, UAV, FUAVs, SAMs, RADARs, objects):
        for key, value in objects.items():
            tag = value[0]
            num = int(value[-1])
            if key == 1:
                if tag == "U":
                    en_att = UAV.unsqueeze(1)
                elif tag == "F":
                    en_att = FUAVs[:, num - 1].unsqueeze(1)
                elif tag == "R":
                    en_att = RADARs[:, num - 1].unsqueeze(1)
                else:
                    en_att = SAMs[:, num - 1].unsqueeze(1)
            else:
                if tag == "U":
                    en_att = torch.cat((en_att, UAV.unsqueeze(1)), dim=1)
                elif tag == "F":
                    en_att = torch.cat((en_att, FUAVs[:, num - 1].unsqueeze(1)), dim=1)
                elif tag == "R":
                    en_att = torch.cat((en_att, RADARs[:, num - 1].unsqueeze(1)), dim=1)
                else:
                    en_att = torch.cat((en_att, SAMs[:, num - 1].unsqueeze(1)), dim=1)
        return en_att

    def get_label(self, P):
        num = len(P[0])
        label = P[:, num - self.output_length:num].unsqueeze(-1)
        return label

    def __read_data__(self):
        en = engine(self.n_UAV, self.n_SAM, self.n_RADAR, self.table_name, self.start_n, self.end_n, self.seq_len,
                    self.n_en_att)
        UAVs, SAMs, RADARs, P_lives = en.get_data(self.norm_max_min, self.con)
        if self.len_slice:
            even_indices = slice(None,None,self.len_slice)
            UAVs = UAVs[:,:,even_indices,:]
            SAMs = SAMs[:,:,even_indices,:]
            RADARs= RADARs[:,:,even_indices,:]
            P_lives= P_lives[:,:,even_indices]
        UAV, FUAV, SAMS, RADARs, P = self.data_integra(torch.tensor(UAVs), torch.tensor(SAMs), torch.tensor(RADARs),
                                                       torch.tensor(P_lives))
        self.en_att_1 = self.get_entities(UAV, FUAV, SAMS, RADARs, self.objects_1)
        self.en_att_2 = self.get_entities(UAV, FUAV, SAMS, RADARs, self.objects_2)
        self.en_att_3 = self.get_entities(UAV, FUAV, SAMS, RADARs, self.objects_3)
        self.en_att_4 = self.get_entities(UAV, FUAV, SAMS, RADARs, self.objects_4)
        self.label = self.get_label(P) * 10

    def __getitem__(self, index):
        '''
        :param en_att_1: [n_entities_1l,seq_len,n_en_att]
        :param en_att_2: [n_entities_2l,seq_len,n_en_att]
        :param en_att_3: [n_entities_3l,seq_len,n_en_att]
        :param en_att_4: [n_entities_4l,seq_len,n_en_att]
        '''
        en_att_1_one = self.en_att_1[index]
        en_att_2_one = self.en_att_2[index]
        en_att_3_one = self.en_att_3[index]
        en_att_4_one = self.en_att_4[index]
        label_one = self.label[index]

        return en_att_1_one, en_att_2_one, en_att_3_one, en_att_4_one, label_one

    def __len__(self):
        return len(self.label)