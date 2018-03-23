import json
import os

class SIM_DATA(object):

    # Only filenames are read. The rest is taken from json file
    def __init__(self,filename="sim_config.json",read=True):
        self.filename = filename
        self.data=[]

        if (read==True):
            self.config_read()
        else:
            # These are default values.
            self.data= {'ENVIROMENT':{'ch_rate'    :300E3,
                                      'temperature':300},
                        'TOPOLOGY':{'n_rings':1,
                                    'n_detectors':2,
                                    'n_faces':2,
                                    'n_sipm_face':64,
                                    'n_asics':8},
                        'TOFPET':{'n_channels':64,
                                  'outlink_rate':(2.6E9/80)/2,
                                  'IN_FIFO_depth':4,
                                  'OUT_FIFO_depth':64*4,
                                  'MAX_WILKINSON_LATENCY':5120,
                                  'TE':7,
                                  'TGAIN':1}
                       }
        self.config_write()

    def config_write(self):
        writeName = self.filename
        try:
            with open(writeName,'w') as outfile:
                json.dump(self.data, outfile, indent=4, sort_keys=False)
        except IOError as e:
            print(e)

    def config_read(self):
        try:
            with open(self.filename,'r') as infile:
                self.data = json.load(infile)
        except IOError as e:
            print(e)

if __name__ == '__main__':
    SIM=SIM_DATA(read=False)
    print SIM.data['TOFPET']['TE']
