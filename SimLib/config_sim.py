import json
import os
import numpy as np
import sys
import pandas as pd




class SIM_DATA(object):

    # Only filenames are read. The rest is taken from json file
    def __init__(self,filename="sim_config.json",read=True):
        self.filename = filename
        self.data=[]

        if (read==True):
            self.config_read()
        else:
            # These are default values.
            # L1 output data frame = QDC[10] + TDC[10] + SiPM[20] = 40 bits
            self.data= {'ENVIRONMENT'  :{'ch_rate'     :1500E3,
                                        'temperature' :300,
                                        'path_to_files': "/home/viherbos/DAQ_DATA/NEUTRINOS/CONT_RING/",
                                        'file_name': "p_FR_infinity_",
                                        'out_file_name':"daq_output",
                                        'n_files' : 8,
                                        'n_events': 25000},

                        'SIPM'        :{'size'        :[1,3,3]},

                        'TOPOLOGY'    :{'radius_int'   :1000,
                                        'radius_ext'   :1000,
                                        'sipm_int_row':178,
                                        'sipm_ext_row':232,
                                        'n_rows'      :16},

                        'TOFPET'      :{'n_channels'  :64,
                                        'outlink_rate': (2.6E9/80)/2.0,
                                        # 80 bits per TOFPET output frame
                                        'IN_FIFO_depth':4,
                                        'OUT_FIFO_depth':64*4,
                                        'MAX_WILKINSON_LATENCY':5120,
                                        'TE':1,
                                        'TGAIN':1},

                        'L1'          :{'L1_outrate'    :700E6,
                                        'frame_process' :10E6,
                                        'FIFO_L1a_depth':512,
                                        'FIFO_L1b_depth':256,
                                        'buffer_size'   :480,
                                        'n_asics'       :12,
                                        'n_L1'          :10000,
                                        'TE'            :4 }
                       }

    def config_write(self):
        writeName = self.filename
        try:
            with open(writeName,'w') as outfile:
                json.dump(self.data, outfile, indent=4, sort_keys=False)
                print self.data
        except IOError as e:
            print(e)

    def config_read(self):
        try:
            with open(self.filename,'r') as infile:
                self.data = json.load(infile)
                print self.data
        except IOError as e:
            print(e)



if __name__ == '__main__':

    filename = "R7"
    SIM=SIM_DATA(filename = "/home/viherbos/DAQ_DATA/NEUTRINOS/CONT_RING/"+filename+".json",
                 read = False)
    SIM.config_write()
