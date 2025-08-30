import torch.nn as nn
import torch
import torch.nn.functional as F

class Relation (nn.Module):
    '''relation embedding'''
    def  __init__(self, input_size, effect_size,nlayers, hidden_size):
        super(Relation, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=nlayers, batch_first = False)
        self.linear_1 = torch.nn.Linear(hidden_size, hidden_size)
        self.linear_2 = torch.nn.Linear(hidden_size, effect_size)
    def forward(self, x):
        '''x1: [n_entities, seq_len, batch_size, input_size]'''
        out,_ = self.gru(x)
        out = self.linear_1(out)
        out = F.relu(out)
        out = F.dropout(out, p=0.2)
        out = self.linear_2(out)
        return out

class Evaluation(nn.Module):
    def __init__(self, effect_size, output_size, nlayers,  hidden_size, length):
        super(Evaluation, self).__init__()
        self.gru = nn.GRU(input_size=effect_size, hidden_size=hidden_size, num_layers=nlayers, batch_first = False)
        self.linear_1 = torch.nn.Linear(hidden_size, hidden_size)
        self.linear_2 = torch.nn.Linear(hidden_size, output_size)
        self.length = length
    def forward(self, x):
        '''x: [seq_len, batch_size, input_size]'''
        out,_  = self.gru(x)
        out = self.linear_1(_)
        out = F.relu(out)
        out = F.dropout(out, p=0.2)
        out = self.linear_2(out)
        return out

class Model_layer(nn.Module):
    def __init__(self,input_size,effect_size,output_size,output_length,nlayers_1,nlayers_2, hidden_size_1 = 60, hidden_size_2 = 60):
        super(Model_layer, self).__init__()
        self.effect_size = effect_size
        self.relation_1 = Relation(input_size, effect_size, nlayers_1, hidden_size_1)
        self.relation_2 = Relation(input_size, effect_size, nlayers_1, hidden_size_1)
        self.relation_3 = Relation(input_size, effect_size, nlayers_1, hidden_size_1)
        self.r_type_dict = {1: self.relation_1, 2: self.relation_1, 3: self.relation_3}
        self.evaluation = Evaluation(effect_size, output_size, nlayers_2, hidden_size_2,output_length)

    def m_interacetion_seg(self,en_att_l,en_att_u,rs,rr,effect):
        '''
        :param en_att_l: [n_entities_lower,seq_len,batch_size,n_en_att]
        :param en_att_u: [n_entities_upper,seq_len,batch_size,n_en_att]
        :param rs: [n_relations_lower-upper,seq_len,batch_size,n_entities_lower]
        :param rr: [n_relations_lower-upper,seq_len,batch_size,n_entities_upper]
        :param effect: [n_entities_lower,seq_len,batch_size,n_effect_att]
        :return: x: [n_relations_lower-upper,seq_len,batch_size,2*n_en_att+n_effect_att]
        '''
        en_att_l_new =  torch.cat((effect, en_att_l), 3).transpose(0,2)
        en_att_u_new = en_att_u.transpose(0,2)
        rs_new = rs.transpose(0,2)
        rr_new = rr.transpose(0,2)
        M_s = torch.matmul(rs_new, en_att_l_new)
        M_r = torch.matmul(rr_new, en_att_u_new)
        x = torch.cat([M_r,M_s], axis=3).transpose(0,2)
        return x

    def effect_cal(self, x, r_type, seq_len, batch_size):
        '''
        :param x: [n_relations_entity,seq_len,batch_size,2*n_en_att+n_effect_att]
        :param r_type: [n_entities_upper,2,n_r_type]
        :param effect: [n_entities_lower,seq_len,batch_size,n_effect_att]
        :return: x: [n_relations_lower-upper,seq_len,batch_size,2*n_en_att+n_effect_att]
        '''
        start = 0
        for a, entity in enumerate(r_type):
            effect_one = torch.zeros(seq_len, batch_size, self.effect_size)
            for i, j in zip( entity[1],entity[0]):
                for one in x[start:start + i]:
                    effect_one += self.r_type_dict[j](one)
                start += i
            if a == 0:
                effect = effect_one.unsqueeze(0)
            else:
                effect = torch.cat((effect, effect_one.unsqueeze(0)), 0)
        return effect

    def forward(self,en_att_1,en_att_2,en_att_3,rs_12,rr_12,rs_23,rr_23,r_type_12,r_type_23):
        '''
        :param en_att_1: [n_entities_1l,seq_len,batch_size,n_en_att]
        :param en_att_2: [n_entities_2l,seq_len,batch_size,n_en_att]
        :param en_att_3: [n_entities_3l,seq_len,batch_size,n_en_att]
        :param rs_12: [n_relations_1l-2l,seq_len,batch_size,n_entities_1l]
        :param rs_23: [n_relations_2l-3l,seq_len,batch_size,n_entities_2l]
        :param rr_12: [n_relations_1l-2l,seq_len,batch_size,n_entities_2l]
        :param rr_23: [n_relations_2l-3l,seq_len,batch_size,n_entities_3l]
        :param r_type_12: [n_entities_2l,2,n_r_type]
        :param r_type_23: [n_entities_3l,2,n_r_type]
        :return:
         out: [out_seq_len,batch_size, output]
        '''
        en_att_1 = torch.permute(en_att_1, (1 ,2 ,0, 3))
        en_att_2 = torch.permute(en_att_2, (1, 2, 0, 3))
        en_att_3 = torch.permute(en_att_3, (1, 2, 0, 3))
        n_entities_1l, seq_len, batch_size, n_en_att = en_att_1.shape
        effect1 = torch.zeros(n_entities_1l, seq_len, batch_size, self.effect_size)
        x1 = self.m_interacetion_seg(en_att_1,en_att_2,rs_12,rr_12,effect1)
        effect2 = self.effect_cal(x1, r_type_12, seq_len, batch_size)
        x2 = self.m_interacetion_seg(en_att_2, en_att_3, rs_23, rr_23, effect2)
        effect3 = self.effect_cal(x2, r_type_23, seq_len, batch_size)
        out = self.evaluation(effect3.reshape(-1,batch_size,self.effect_size))
        return out







