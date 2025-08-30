import mysql.connector as mc
from mysql.connector import errorcode
import json
import numpy as np
import copy

config = {
  'user': 'root',
  'password': ' ',
  'host': '127.0.0.1',
  'database': 'env',
  'raise_on_warnings': True,
}

class engine:
    def __init__(self,n_UAV, n_SAM, n_RADAR, Table_name, start_n, end_n, seq_len,n_en_att):
        self.n_UAV = n_UAV
        self.n_SAM = n_SAM
        self.n_RADAR = n_RADAR
        self.Table_name = Table_name
        self.start_n = start_n
        self.end_n = end_n
        self.seq_len = seq_len
        self.n_att = n_en_att
        self.connect()

    def connect(self):
        try:
            self.cnx = mc.connect(**config)
        except mc.Error as err:
            if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
                print("用户名密码错误")
            elif err.errno == errorcode.ER_BAD_DB_ERROR:
                print("数据库不存在")
            else:
                print(err)
        else:
            print('线程数据库{}已连接'.format(config['database']))
            self.cursor = self.cnx.cursor()

    def normalize(self,data):
        column_max = np.max(data, axis= 0)
        column_min = np.min(data, axis= 0)
        return [list(column_max), list(column_min)]

    def column_normalize(self,data,norm):
        norm_factor = np.array(norm[0]) -  np.array(norm[1])
        for i, factor in enumerate(norm_factor):
            if int(factor) == 0:
                norm_factor[i] = 0.01
        try:
            normalized_data = (data-np.array(norm[1]))/norm_factor
        except:
            print(np.array(data).shape)
            print(np.array(norm).shape)
        return normalized_data

    def get_norm(self):
        sql = "SELECT Type,X,Y,ATT_one,ATT_two,P_live FROM %s WHERE Step <= %f" % (self.Table_name, self.seq_len)
        self.cursor.execute(sql)
        results = self.cursor.fetchall()
        UAVs = []
        SAMs = []
        Radars = []
        Plives = []
        for row in results:
            if row[0][0] == "U":
                UAVs.append([row[1], row[2], row[3], row[4]])
                Plives.append(row[5])
            elif row[0][0] == "S":
                SAMs.append([row[1], row[2], row[3], row[4]])
            else:
                Radars.append([row[1], row[2], row[3], row[4]])
        UAV_max_min = self.normalize(np.array(UAVs, dtype=np.float32))
        SAM_max_min = self.normalize(np.array(SAMs, dtype=np.float32))
        if self.n_RADAR == 0: RADAR_max_min = None
        else: RADAR_max_min = self.normalize(np.array(Radars,dtype=np.float32))
        return  UAV_max_min,SAM_max_min,RADAR_max_min

    def get_data(self,norm_max_min,con):
        if self.n_RADAR == 0:
            n_RADAR = self.n_SAM
        else:
            n_RADAR = self.n_RADAR
        if con.seq_len_sum:
            start_n = self.start_n * (self.n_UAV + n_RADAR + self.n_SAM) *  con.seq_len_sum + 1
            end_n = self.end_n * (self.n_UAV + n_RADAR + self.n_SAM) *  con.seq_len_sum
        else:
            start_n = self.start_n*(self.n_UAV+n_RADAR+self.n_SAM)*self.seq_len+1
            end_n = self.end_n*(self.n_UAV+n_RADAR+self.n_SAM)*self.seq_len
        sql = "SELECT Type,X,Y,ATT_one,ATT_two,P_live FROM %s WHERE Step <= %f and %f <= ID and ID  <= %f" % (self.Table_name,self.seq_len,start_n, end_n)
        self.cursor.execute(sql)
        results = self.cursor.fetchall()
        UAVs = []
        SAMs = []
        Radars = []
        Plives = []
        for row in results:
            if row[0][0] == "U":
                UAVs.append([row[1], row[2], row[3], row[4]])
                Plives.append(row[5])
            elif row[0][0] == "S":
                SAMs.append([row[1], row[2], row[3], row[4]])
            else:
                Radars.append([row[1], row[2], row[3], row[4]])
        if norm_max_min:
            if not con.UAV_max_min:
               con.UAV_max_min, con.SAM_max_min, con.RADAR_max_min = self.get_norm()
            UAVs= self.column_normalize(np.array(UAVs,dtype=np.float32),con.UAV_max_min)
            SAMs = self.column_normalize(np.array(SAMs,dtype=np.float32),con.SAM_max_min)
            if self.n_RADAR == 0: Radars = None
            else: Radars = self.column_normalize(np.array(Radars,dtype=np.float32),con.RADAR_max_min )
        else:
            UAVs = np.array(UAVs,dtype=np.float32)
            SAMs = np.array(SAMs,dtype=np.float32)
            if self.n_RADAR == 0: Radars = None
            else: Radars = np.array(Radars,dtype=np.float32)
        UAV_seqs =UAVs.reshape(-1,  self.seq_len, self.n_UAV, self.n_att).transpose(0,2,1,3)
        SAM_seqs  = SAMs.reshape( -1,  self.seq_len, self.n_SAM,self.n_att).transpose(0,2,1,3)
        if self.n_RADAR == 0: Radar_seqs = None
        else: Radar_seqs = Radars.reshape( -1, self.seq_len, self.n_RADAR, self.n_att).transpose(0,2,1,3)
        P_lives = np.array(Plives,dtype=np.float32).reshape( -1, self.seq_len, self.n_UAV).transpose(0,2,1)

        return UAV_seqs, SAM_seqs, Radar_seqs, P_lives
