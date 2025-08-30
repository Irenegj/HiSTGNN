import numpy as np
import torch
class configforU_W():
    def __init__(self):
        '''Quantity of Each Entity'''
        self.num_UAV = 5
        self.num_SAM = 3
        self.num_RADAR = 0

        self.table_name = 'env_test_01m_15d_10h_47m_51s'
        self.model = 'HiSTGNN_U_W'

        '''Relationship Description: "WU" and "SU"："Weapon-to-UAV", "US"："UAV-to-Weapon", "UU"："UAV-to-UAV"'''
        self.seq_len_sum = None
        self.relations = { "SU" : 1, "US" : 2, "UU" : 3}
        self.objects_num = {"UAV": 1, "FUAV": self.num_SAM-1, "SAM":self.num_SAM}

        "Hypergraph Representation"
        # Representing the hypernodes from left to right in each layer."
        self.supobjects_1 = ["SAM", "UAV", "FUAV"]
        # Representing the receiving hypernodes in each layer and their connected relationship types from left to right.
        self.suprelations_type_12 = {"FUAV": ["SU","UU"], "SAM": ["US"]}
        self.supobjects_2 = ["FUAV", "SAM"]
        self.suprelations_type_23 = {"UAV": ["UU", "SU"]}
        self.supobjects_3 = ["UAV"]

        "Representing the entity types per layer, with the order of entity types strictly following the left-to-right sequence in the graph."
        self.objects_1 = self.compute_object(self.objects_num, self.supobjects_1)
        self.objects_2 = self.compute_object(self.objects_num, self.supobjects_2)
        self.objects_3 = self.compute_object(self.objects_num, self.supobjects_3)

        "Representing the sending entities and receiving entities in the relationship between two layers, with the entity order strictly following the left-to-right sequence in the graph."
        self.relations_s_12, self.relations_r_12 = self.compute_relations(self.objects_1 ,self.objects_2,
                                                                          self.suprelations_type_12)
        self.relations_s_23, self.relations_r_23 = self.compute_relations(self.objects_2, self.objects_3,
                                                                          self.suprelations_type_23)

        "Representing the number of each relationship type connected to the receiving entity , strictly following the left-to-right sequence in the graph."
        self.relations_type_12 = self.compute_sequential_dict(self.objects_2 , self.objects_1 , self.relations_r_12,
                                                              self.suprelations_type_12)
        self.relations_type_23 = self.compute_sequential_dict(self.objects_3, self.objects_2, self.relations_r_23,
                                                              self.suprelations_type_23)

        self.patience = 10
        self.UAV_max_min = None
        self.SAM_max_min = None
        self.RADAR_max_min = None

    def compute_object(self, objects_num, supobjects):
        tag = 1
        objects = {}
        for object_type in supobjects:
            for num in range(objects_num[object_type]):
                objects[tag] = object_type + str(num+1)
                tag += 1
        return objects

    def compute_relations(self, objects_s, objects_r, suprelations_type):
        relations_s = []
        relations_r = []
        for object_r_type,  r_type_list in suprelations_type.items():
                for index_r, object_tag_r in objects_r.items():
                    if object_r_type in object_tag_r:
                        for r_type in r_type_list:
                            for index_s, object_tag_s  in objects_s.items():
                                if (r_type[0] in object_tag_s) and (object_r_type != object_tag_s[:-1]):
                                    relations_r.append(index_r)
                                    relations_s.append(index_s)
        return relations_s, relations_r

    def compute_sequential_dict(self, object_r, object_s, relation_r, suprelations_type):
        sequential_dict = {}
        unique_relation_r_lst = list(set(relation_r))
        for index, num in enumerate(unique_relation_r_lst):
            object_r_type = object_r[num][:-1]
            if object_r_type in suprelations_type:
                num_sequentia_list = [0]*len(self.relations)
                sequential_list = self.compute_sequential_list(suprelations_type[object_r_type])
                for index_r_type, r_type in enumerate(suprelations_type[object_r_type]):
                    for index_s, object_tag_s in object_s.items():
                        if (r_type[0] in object_tag_s) and (object_r_type != object_tag_s[:-1]):
                            num_sequentia_list [index_r_type] += 1
                sequential_dict[index+1] = {"type": sequential_list, "num": num_sequentia_list}
        return sequential_dict

    def compute_sequential_list(self,r_type_list):
        sequential_list = []
        for type in r_type_list:
            sequential_list.append(self.relations[type])
        if len(sequential_list) < len(self.relations):
            for i in range(len(self.relations)):
                if i + 1 not in sequential_list:
                    sequential_list.append(i + 1)
        return sequential_list

class configforU_W_R():
    def __init__(self):
        '''Quantity of Each Entity'''
        self.num_UAV = 5
        self.num_SAM = 3
        self.num_RADAR = 2
        self.table_name = "env_test_12M_03D_09H_41m_09s"
        self.model = 'HiSTGNN_U_W_R'

        '''Relationship Description: "WU" and "SU"："Weapon-to-UAV", "US"："UAV-to-Weapon", "UU"："UAV-to-UAV"'''
        self.relations = { "UU" : 1, "SU" : 2, "US" : 3, "RS" : 4, "UR" : 5}
        self.objects_num = {"UAV": 1, "FUAV": self.num_UAV-1, "SAM":self.num_SAM, "RADAR":self.num_RADAR}
        self.seq_len_sum = None

        "Hypergraph Representation"
        #Representing the hypernodes from left to right in each layer."
        self.supobjects_1 = ["UAV", "FUAV"]
        #Representing the receiving hypernodes in each layer and their connected relationship types from left to right.
        self.suprelations_type_12  = { "RADAR": ["UR"]}
        self.supobjects_2 = ["SAM", "UAV", "FUAV", "RADAR"]
        self.suprelations_type_23 = {"FUAV": ["SU","UU"], "SAM": ["US", "RS"]}
        self.supobjects_3 = ["FUAV", "SAM"]
        self.suprelations_type_34 = {"UAV": ["UU", "SU"]}
        self.supobjects_4 = ["UAV"]

        "Representing the entity types per layer, with the order of entity types strictly following the left-to-right sequence in the graph."
        self.objects_1 = self.compute_object(self.objects_num, self.supobjects_1)
        self.objects_2 = self.compute_object(self.objects_num, self.supobjects_2)
        self.objects_3 = self.compute_object(self.objects_num, self.supobjects_3)
        self.objects_4 = self.compute_object(self.objects_num, self.supobjects_4)

        "Representing the sending entities and receiving entities in the relationship between two layers, with the entity order strictly following the left-to-right sequence in the graph."
        self.relations_s_12, self.relations_r_12 = self.compute_relations(self.objects_1 ,self.objects_2,
                                                                          self.suprelations_type_12)
        self.relations_s_23, self.relations_r_23 = self.compute_relations(self.objects_2, self.objects_3,
                                                                          self.suprelations_type_23)
        self.relations_s_34, self.relations_r_34 = self.compute_relations(self.objects_3, self.objects_4,
                                                                          self.suprelations_type_34)

        "Representing the number of each relationship type connected to the receiving entity , strictly following the left-to-right sequence in the graph."
        self.relations_type_12 = self.compute_sequential_dict(self.objects_2 , self.objects_1 , self.relations_r_12,
                                                              self.suprelations_type_12)
        self.relations_type_23 = self.compute_sequential_dict(self.objects_3, self.objects_2, self.relations_r_23,
                                                              self.suprelations_type_23)
        self.relations_type_34 = self.compute_sequential_dict(self.objects_4, self.objects_3, self.relations_r_34,
                                                              self.suprelations_type_34)

        self.patience = 10
        self.UAV_max_min = None
        self.SAM_max_min = None
        self.RADAR_max_min = None

    def compute_object(self, objects_num, supobjects):
        tag = 1
        objects = {}
        for object_type in supobjects:
            for num in range(objects_num[object_type]):
                objects[tag] = object_type + str(num+1)
                tag += 1
        return objects

    def compute_relations(self, objects_s, objects_r, suprelations_type):
        relations_s = []
        relations_r = []
        for object_r_type,  r_type_list in suprelations_type.items():
                for index_r, object_tag_r in objects_r.items():
                    if object_r_type in object_tag_r:
                        for r_type in r_type_list:
                            for index_s, object_tag_s  in objects_s.items():
                                if (r_type[0] in object_tag_s) and (object_r_type != object_tag_s[:-1]):
                                    relations_r.append(index_r)
                                    relations_s.append(index_s)
        return relations_s, relations_r

    def compute_sequential_dict(self, object_r, object_s, relation_r, suprelations_type):
        sequential_dict = {}
        unique_relation_r_lst = list(set(relation_r))
        for index, num in enumerate(unique_relation_r_lst):
            object_r_type = object_r[num][:-1]
            if object_r_type in suprelations_type:
                num_sequentia_list = [0]*len(self.relations)
                sequential_list = self.compute_sequential_list(suprelations_type[object_r_type])
                for index_r_type, r_type in enumerate(suprelations_type[object_r_type]):
                    for index_s, object_tag_s in object_s.items():
                        if (r_type[0] in object_tag_s) and (object_r_type != object_tag_s[:-1]):
                            num_sequentia_list [index_r_type] += 1
                sequential_dict[index+1] = {"type": sequential_list, "num": num_sequentia_list}
        return sequential_dict

    def compute_sequential_list(self,r_type_list):
        sequential_list = []
        for type in r_type_list:
            sequential_list.append(self.relations[type])
        if len(sequential_list) < len(self.relations):
            for i in range(len(self.relations)):
                if i + 1 not in sequential_list:
                    sequential_list.append(i + 1)
        return sequential_list




