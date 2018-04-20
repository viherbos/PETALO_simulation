import simpy
import random
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from simpy.events import AnyOf, AllOf, Event
import sys
sys.path.append("../PETALO_analysis/")
import fit_library
import HF_translator as HFT
import os
import multiprocessing as mp
from functools import partial
from SimLib import DAQ_infinity as DAQ
from SimLib import HF_files as HF
import time
from SimLib import config_sim as CFG
from SimLib import pet_graphics as PG
import pandas as pd
import math


def DAQ_sch(sim_info):

    param = sim_info['Param']
    env = simpy.Environment()

    start_time = time.time()

    # Work out number of SiPMs based on geometry data
    n_sipms_I = param.P['TOPOLOGY']['sipm_int_row']*param.P['TOPOLOGY']['n_rows']
    n_sipms_O = param.P['TOPOLOGY']['sipm_ext_row']*param.P['TOPOLOGY']['n_rows']
    n_sipms     = n_sipms_I + n_sipms_O

    # Number of ASICs calculation: Inner Face + Outer Face // full + partial
    n_asics_I = int(math.ceil(float(n_sipms_I) / float(param.P['TOFPET']['n_channels'])))
    n_asics_f_I  = n_sipms_I // param.P['TOFPET']['n_channels']  # Fully used
    n_asics_p_I = n_asics_I - n_asics_f_I                       # Partially used
    n_asics_O = int(math.ceil(float(n_sipms_O) / float(param.P['TOFPET']['n_channels'])))
    n_asics_f_O  = n_sipms_O // param.P['TOFPET']['n_channels']
    n_asics_p_O   = n_asics_O - n_asics_f_O      # Number of not fully used ASICs (0 or 1)
    n_asics = n_asics_I + n_asics_O

    # L1 are required with max number of ASICs in param.P['L1']['n_asics']
    # // full + part
    n_L1 = int(math.ceil(float(n_asics) / float(param.P['L1']['n_asics'])))
    n_L1_f = n_asics // param.P['L1']['n_asics']
    n_L1_p = n_L1 - n_L1_f


    print ("Number of SiPM : %d \nNumber of ASICS : %d " % (n_sipms,n_asics))
    print ("Number of L1 : %d " % (n_L1))


    SiMP_Matrix_I = np.reshape(np.arange(0,n_sipms_I),
                                (param.P['TOPOLOGY']['n_rows'],
                                param.P['TOPOLOGY']['sipm_int_row']))
    SiMP_Matrix_O = np.reshape(np.arange(n_sipms_I,n_sipms),
                                (param.P['TOPOLOGY']['n_rows'],
                                param.P['TOPOLOGY']['sipm_ext_row']))
    # SiPM matrixs Inner face and Outer face



    #//////////////////////////////////////////////////////////////////////////
    #////                       SYSTEM INSTANCIATION                       ////
    #//////////////////////////////////////////////////////////////////////////

    data_out = [[] for i in range(n_L1)]

    L1    = [ DAQ.L1( env        = env,
                      out_stream = data_out[i],
                      param      = param,
                      L1_id      = i)
              for i in range(n_L1_f)]
    if (n_L1_p == 1):
        L1.append(DAQ.L1( env        = env,
                          out_stream = data_out[n_L1_f],
                          param      = param,
                          L1_id      = n_L1_f))

    block_size = param.P['TOFPET']['n_channels']

    ASICS_I = [DAQ.FE_asic(env     = env,
                         param   = param,
                         data    = DATA[:,np.reshape(SiMP_Matrix_I[:,i*4:(i+1)*4],-1)],
                         timing  = sim_info['timing'],
                         sensors = sim_info['Param'].sensors[np.reshape(SiMP_Matrix_I[:,i*4:(i+1)*4],-1)],
                         asic_id = i)
             for i in range(n_asics_f_I)]
    if (n_asics_p_I == 1):
        ASICS_I.append(DAQ.FE_asic(env     = env,
                                   param   = param,
                                   data    = DATA[:,np.reshape(SiMP_Matrix_I[:,n_asics_f_I*4:],-1)],
                                   timing  = sim_info['timing'],
                                   sensors = sim_info['Param'].sensors[np.reshape(SiMP_Matrix_I[:,n_asics_f_I*4:],-1)],
                                   asic_id = n_asics_f_I))

    ASICS_O = [DAQ.FE_asic(env     = env,
                         param   = param,
                         data    = DATA[:,np.reshape(SiMP_Matrix_O[:,i*4:(i+1)*4],-1)],
                         timing  = sim_info['timing'],
                         sensors = sim_info['Param'].sensors[np.reshape(SiMP_Matrix_O[:,i*4:(i+1)*4],-1)],
                         asic_id = i)
             for i in range(n_asics_f_O)]
    if (n_asics_p_O == 1):
        ASICS_O.append(DAQ.FE_asic(env     = env,
                                   param   = param,
                                   data    = DATA[:,np.reshape(SiMP_Matrix_O[:,n_asics_f_O*4:],-1)],
                                   timing  = sim_info['timing'],
                                   sensors = sim_info['Param'].sensors[np.reshape(SiMP_Matrix_O[:,n_asics_f_O*4:],-1)],
                                   asic_id = n_asics_f_O))
    # for i in range(len(ASICS_I)):
    #     print("ASIC %d" % (i))
    #     print sim_info['Param'].sensors[np.reshape(SiMP_Matrix_I[:,i*4:(i+1)*4],-1)]

    ASICS_I.extend(ASICS_O)
    ASICS = ASICS_I

    # Wiring of ASICS to L1
    j,count = 0,0

    for i in range(len(ASICS)):
        ASICS[i].Link.out = L1[j]
        if (count < param.P['L1']['n_asics']-1):
            count += 1
        else:
            j += 1
            count = 0

    # Run Simulation for a very long time to force flush of FIFOs
    env.run(until = 100E9)

    # Gather output information and statistics
    OUTPUT_L1     = [L1[i]() for i in range(n_L1)]
    OUTPUT_ASICS  = [ASICS[i]() for i in range(n_asics)]

    elapsed_time = time.time()-start_time


    topology = {'n_sipms_I':n_sipms_I, 'n_sipms_O':n_sipms_O, 'n_sipms': n_sipms,
                'n_asics_I':n_asics_I, 'n_asics_f_I':n_asics_f_I,'n_asics_p_I':n_asics_p_I,
                'n_asics_O':n_asics_O, 'n_asics_f_O':n_asics_f_O,'n_asics_p_O':n_asics_p_O,
                'n_asics':n_asics, 'n_L1':n_L1, 'n_L1_f':n_L1_f, 'n_L1_p':n_L1_p}
    print ("IT TOOK %d SECONDS TO DO THIS" % elapsed_time)
    print ("Number of L1 Inst : %d \nNumber of ASICS Inst : %d " % (len(L1),len(ASICS)))


    return {'L1_out':OUTPUT_L1, 'ASICS_out':OUTPUT_ASICS}, topology



def DAQ_OUTPUT_processing(SIM_OUT,n_L1):
    data, in_time, out_time, lostL1a, lostL1b = [],[],[],[],[]
    SIM_OUT_L1      = np.array(SIM_OUT['L1_out'])
    SIM_OUT_ASICs   = np.array(SIM_OUT['ASICS_out'])

    # Gather information from L1
    for j in range(n_L1):
        lostL1a.append(SIM_OUT_L1[j]['lostL1a'])
        lostL1b.append(SIM_OUT_L1[j]['lostL1b'])
        for i in range(len(SIM_OUT_L1[j]['data_out'])):
            #if SIM_OUT[j]['data_out'][i]['data'][0] > 0:
            data.append(SIM_OUT_L1[j]['data_out'][i]['data'])
            in_time.append(SIM_OUT_L1[j]['data_out'][i]['in_time'])
            out_time.append(SIM_OUT_L1[j]['data_out'][i]['out_time'])


    A = np.array(data)
    sort = np.array([i[1] for i in A])
    A = A[np.argsort(sort)]

    n_TDC = np.array([])
    i_TDC = np.array([])
    TDC = np.array([A[i][1] for i in range(len(A))])


    prev=0
    for i in TDC:
        if (i != prev):
            cond = np.array((TDC == i))
            n_TDC = np.concatenate((n_TDC,[np.sum(cond)]),axis=0)
            i_TDC = np.concatenate((i_TDC,[i]),axis=0)
            prev = i

    print i_TDC.shape

    event = 0
    A_index = 0

    data = np.zeros((n_events,n_sipms),dtype='int32')
    for i in i_TDC:
        for j in range(int(n_TDC[event])):
            for l in range(int(A[A_index][0])):
                data[event,int(A[A_index][2*l+2])-1000] = A[A_index][2*l+3]

            A_index += 1

        event += 1

    output = {'data': data,
              'L1': {'in_time': in_time, 'out_time': out_time,
                           'lostL1a': lostL1a, 'lostL1b': lostL1b}
            }


    return output

# def DAQ_sim(L1h_array, sim_info):
#
#     kargs = {'sim_info':sim_info}
#     DAQ_map = partial(DAQ_gen, **kargs)
#
#     # Multiprocess Work
#     # pool_size = mp.cpu_count()
#     # pool = mp.Pool(processes=pool_size)
#     #
#     # pool_output = pool.map(DAQ_map, [[i] for i in L1h_array])
#     #
#     # pool.close()
#     # pool.join()
#     pool_output = DAQ_map([1])
#
#     return pool_output



if __name__ == '__main__':

    CG = CFG.SIM_DATA(filename = "sim_config.json",
                          read = True)
    CG = CG.data
    # Read data from json file
    n_sipms_int = CG['TOPOLOGY']['sipm_int_row']*CG['TOPOLOGY']['n_rows']
    n_sipms_ext = CG['TOPOLOGY']['sipm_ext_row']*CG['TOPOLOGY']['n_rows']
    n_sipms     = n_sipms_int + n_sipms_ext

    n_files = 1
    #Number of files to group for data input
    A = HF.hdf_compose( "/home/viherbos/DAQ_DATA/NEUTRINOS/CONT_RING/",
                        "p_FR_infinity_",
                        range(n_files),n_sipms)
    DATA,sensors,n_events = A.compose()

    # Number of events for simulation
    n_events = 200
    DATA = DATA[0:n_events,:]
    print (" %d EVENTS IN %d H5 FILES" % (n_events,n_files))

    # SHOW = PG.DET_SHOW(CG.data)
    # os.chdir("/home/viherbos/DAQ_DATA/NEUTRINOS/RING/")
    # filename = "p_FRSET_0.h5"
    # positions = np.array(pd.read_hdf(filename,key='sensors'))
    # data = np.array(pd.read_hdf(filename,key='MC'), dtype = 'int32')
    # SHOW(positions,data,0,True,False)


    Param = DAQ.parameters(CG,sensors,n_events)

    timing = np.random.randint(0,int(1E9/Param.P['ENVIRONMENT']['ch_rate']),size=n_events)

    # All sensors are given the same timestamp in an events
    sim_info = {'DATA': DATA, 'timing': timing, 'Param': Param }

    # L1s = range(CG.data['L1']['n_L1'])
    # pool_output = DAQ_sim(L1s,sim_info)

    SIM_OUT,topology = DAQ_sch(sim_info)
    print topology

    # Data Output recovery
    out = DAQ_OUTPUT_processing(SIM_OUT,topology['n_L1'])



    # DATA ANALYSIS AND GRAPHS
    latency = np.array(out['L1']['out_time'])-np.array(out['L1']['in_time'])

    print ("LOST DATA L1A -> L1B = %d" % (np.array(out['L1']['lostL1a']).sum()))
    print ("LOST DATA L1B -> OUTPUT = %d" % (np.array(out['L1']['lostL1b']).sum()))

    #print ("A total of %d events processed" % np.array(outlink_ch).sum())

    # latency = np.array([]).reshape(0,1)
    # lostP = np.array([]).reshape(0,1)
    # lostC = np.array([]).reshape(0,1)
    # lostL1 = np.array([]).reshape(0,1)
    # data   = np.array([]).reshape(0,6)
    #
    #
    # for i in L1s:
    #     data_frame_array = pool_output[i][0]['data_out'][:,:]
    #     lostP_aux  = pool_output[i][0]['lostP']
    #     lostC_aux  = pool_output[i][0]['lostC']
    #     lostL1_aux = pool_output[i][0]['lostL1']
    #
    #
    #     in_time_array  = data_frame_array[:,4]
    #     out_time_array = data_frame_array[:,5]
    #     latency_aux  = out_time_array - in_time_array
    #     #latency = np.vstack([latency,latency_aux.reshape(-1,1)])
    #     latency = np.pad(latency,((len(latency_aux),0),(0,0)),
    #                              mode='constant',
    #                              constant_values=0)
    #     latency[0:len(latency_aux),0] = latency_aux
    #
    #     lostP = np.pad(lostP,((len(lostP_aux),0),(0,0)),
    #                              mode='constant',
    #                              constant_values=0)
    #     lostP[0:len(lostP_aux),0] = lostP_aux[:,0]
    #
    #     lostC = np.pad(lostC,((len(lostC_aux),0),(0,0)),
    #                              mode='constant',
    #                              constant_values=0)
    #     lostC[0:len(lostC_aux),0] = lostC_aux[:,0]
    #
    #     lostL1 = np.pad(lostL1,((len(lostL1_aux),0),(0,0)),
    #                              mode='constant',
    #                              constant_values=0)
    #     lostL1[0:len(lostL1_aux),0] = lostL1_aux[:,0]
    #
    #     data = np.pad(data,((data_frame_array.shape[0],0),(0,0)),
    #                              mode='constant',
    #                              constant_values=0)
    #     data[0:data_frame_array.shape[0],:] = data_frame_array
    #
    #
    fit = fit_library.gauss_fit()
    fig = plt.figure(figsize=(16,8))
    # fit(lostP,'sqrt')
    # fit.plot(axis = fig.add_subplot(231),
    #         title = "FE FIFO drops",
    #         xlabel = "Lost Events",
    #         ylabel = "Hits",
    #         res = False)
    # fit(lostC,'sqrt')
    # fit.plot(axis = fig.add_subplot(234),
    #         title = "Data Link FIFO drops",
    #         xlabel = "Lost Events",
    #         ylabel = "Hits",
    #         res = False)
    # fit(lostL1,'sqrt')
    # fit.plot(axis = fig.add_subplot(235),
    #         title = "L1 FIFO drops",
    #         xlabel = "Lost Events",
    #         ylabel = "Hits",
    #         res = False)
    # fit(outlink_ch,'sqrt')
    # fit.plot(axis = fig.add_subplot(232),
    #         title = "Recovered Channel Data",
    #         xlabel = "Ch_Events",
    #         ylabel = "Hits",
    #         res = False)
    fit(latency,50)
    fit.plot(axis = fig.add_subplot(233),
            title = "Data Latency",
            xlabel = "Latency in nanoseconds",
            ylabel = "Hits",
            res = False)
    #
    #
    # fig.add_subplot(236)
    # x_data = fit.bin_centers
    # y_data = np.add.accumulate(fit.hist_fit)/np.max(np.add.accumulate(fit.hist_fit))
    # plt.plot(x_data,y_data)
    # plt.ylim((0.9,1.0))
    #
    fig.tight_layout()

    # Write output to file
    DAQ_dump = HF.DAQ_IO("/home/viherbos/DAQ_DATA/NEUTRINOS/CONT_RING/",
                            "daq_output.h5",
                            "p_FR_infinity_0.h5",
                            "daq_out.h5")
    DAQ_dump.write_out(out['data'],topology)

    plt.show()
    #
    #
