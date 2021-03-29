from brian import *
import sys
import multiprocessing as mp
import time
import numpy as np
import os
import socket
from scipy import sparse
from random import randint

start_time = time.time()
numcores   = mp.cpu_count()
defaultclock.reinit()
defaultclock.dt = 0.1*ms

# get integer from standard input to label output files
par = int(sys.argv[1])

# define function to decode angles from spike counts
def decode(firing_rate, N_e):
    angles = np.arange(0,N_e)*2*np.pi/N_e
    R = np.sum(np.dot(firing_rate,np.exp(1j*angles)))/N_e
    angle = np.angle(R)
    if angle < 0:
        angle +=2*np.pi 
    return angle 

# define function to get bump center diffusion from full rastergram (i,t)
def readout(i, t, sim_time, N_e):
    w1      = 100*ms
    w2      = 250*ms
    n_wins  = int((sim_time-w2)/w1)

    decs = []
    for ti in range(int(n_wins)):
        fr  = np.zeros(N_e)
        idx = ((t>ti*w1-w2/2) & (t<ti*w1+w2/2))
        ii  = i[idx]
        for n in range(N_e):
            fr[n] = sum(ii == n)
        dec = decode(fr, N_e)
        decs.append(dec)

    return decs, n_wins

# define network simulation code for one trial (the i-th trial in simulation batch par)
def run_simulation(i): 

    global par
    log_file    = "simulation_%i_%f_%s_%i" %(os.getpid(), time.time(), socket.gethostname(), i)
    beh_log     = "output_beh%d.txt" %(par)
    bmp_log     = "output_bump%d.txt" %(par)
    irec_log	= "output_irec%d.txt" %(par)
    
    print log_file
    
    # choose whether to generate a new connectivity (loadconnections=False) or load it from drive (faster)
    loadconnections = True #False
    
    # simulation parameters
    nstims=50 # number of possible equidistant locations for the stimulus (will be randomly drawn)
    stim_on=2000*ms
    stim_off=3000*ms
    stimE=0.24*mV # stimulus input
    epsE=60 # stimulus width (deg)
    stimI=0*mV
    epsI=0
    runtime=7*second

    N=20000 # total number of neurons
    K=500 # total number of inputs per neuron
 
    tauE=20*ms # membrane time constant
    tauI=10*ms 
    taua = 3*ms # AMPA synapse decay
    taun = 50*ms # NMDA synapse decay
    taug = 4*ms # GABA synapse decay
    
    # parameters for short-term plasticity
    Ustp = 0.03 ############ CONDITION STP down: 0.04
    taud = 200*ms
    tauf = 450*ms

    Vt  = 20*mV          # spike threshold
    Vr  = -3.33*mV          # reset value
    refE= 0*ms                # refractory periods
    refI= 0*ms                # refractory periods

    # connectivity strengths
    gEEA=533.3*mV*ms  # AMPA E-to-E
    gEEN=490.64*mV*ms  # NMDA E-to-E  ############ CONDITION E->E down: 479.97*mV*ms
    gEIA=67.2*mV*ms  # AMPA E-to-I
    gEIN=7.4*mV*ms  # NMDA E-to-I  ############ CONDITION E->I down: 3.7*mV*ms
    gIE=-138.6*mV*ms # I-to-E
    gII=-90.6*mV*ms # I-to-I
    sigmaEE=30 # E-to-E footprint in degrees
    sigmaEI=35 # E-to-I footprint in degrees
    sigmaIE=30 # I-to-E footprint in degrees
    sigmaII=30 # I-to-I footprint in degrees

    #these are intermediate calculations needed for the equations below

    NE=int(ceil(0.8*N)) # number of excitatory neurons
    NI=int(floor(0.2*N)) # number of inhibitory neurons

    KE=int(ceil(0.8*K)) # number of excitatory inputs
    KI=int(floor(0.2*K)) # number of inhibitory inputs

    sigEE=sigmaEE/360.0
    sigEI=sigmaEI/360.0
    sigIE=sigmaIE/360.0
    sigII=sigmaII/360.0
    epsE = epsE/360.0

    R0 = 20.0/second 
    u_ss = Ustp*(1.+R0*tauf)/(1.+Ustp*R0*tauf)
    u_ss0= 0.03*(1.+R0*tauf)/(1.+0.03*R0*tauf)
    synfact_ss = u_ss/(1.+taud*R0*u_ss)
    synfact_ss0 = u_ss0/(1.+taud*R0*u_ss0)
    
    gEEA=gEEA/sqrt(KE)/taua*synfact_ss0/synfact_ss #
    gEEN=gEEN/sqrt(KE)/taun*synfact_ss0/synfact_ss #
    gEIA=gEIA/sqrt(KE)/taua
    gEIN=gEIN/sqrt(KE)/taun
    gIE=gIE/sqrt(KI)/taug
    gII=gII/sqrt(KI)/taug

    stimE=stimE*sqrt(KE)
    stimI=stimI*sqrt(KE)

    #ring model connectivity
    def conn(k,sig,C):
        if abs(k)<=0.5:
            value=C*exp(-0.5*(k/sig)**2)/sqrt(2*pi)/sig
        else:
            value=C*exp(-0.5*((abs(k)-1)/sig)**2)/sqrt(2*pi)/sig
        return value

    #equations for each neuron. xpre and satura are shadow variables that track the saturated state
    # of a neuron's outgoing NMDA synapses, and this is used as a "synaptic depression" effect on x
    # (see modulation='satura' below) so as to mimic the NMDA saturation that is modeled in 
    # Compte et al., 2000
    eqs = '''
    Irec = gea+gen+gi + Ix: mV
    dV/dt = (Irec-V+Iext)/tau : mV
    dgea/dt = -gea/taua          : mV
    dgen/dt = -gen/taun   : mV
    dgi/dt = -gi/taug             : mV
    tau : second
    Ix : mV
    Iext : mV
    '''

    # generate the two populations
    networkE=NeuronGroup(NE,model=eqs,threshold=Vt,reset=Vr, refractory=refE)
    networkI=NeuronGroup(NI,model=eqs,threshold=Vt,reset=Vr, refractory=refI)
    
    # initialize parameters for the two populations
    networkE.tau=tauE
    networkI.tau=tauI
    networkE.Ix=1.66*sqrt(KE)*mV
    networkI.Ix=1.85*0.83*sqrt(KE)*mV
    networkE.V =  Vr + rand(NE)*(Vt - Vr) # random initial conditions for V in each trial
    networkI.V =  Vr + rand(NI)*(Vt - Vr)
    networkE.Iext=0*mV
    networkI.Iext=0*mV

    if loadconnections:
        # load the connectivity from file (connections_sp.npz)
        loader = np.load('connections_sp.npz', allow_pickle=True)
        WW = sparse.csr_matrix((loader['CEEd'], loader['CEEi'], loader['CEEp']), shape=loader['CEEs'])
        WW[WW != 0] = 1
        C1=Connection(networkE, networkE, 'gea', weight=gEEA*WW)
        C2=Connection(networkE, networkE, 'gen', weight=gEEN*WW)
        WW = sparse.csr_matrix((loader['CEId'], loader['CEIi'], loader['CEIp']), shape=loader['CEIs'])
        WW[WW != 0] = 1
        C3=Connection(networkE, networkI, 'gea', weight=gEIA*WW)
        C4=Connection(networkE, networkI, 'gen', weight=gEIN*WW)
        WW = sparse.csr_matrix((loader['CIEd'], loader['CIEi'], loader['CIEp']), shape=loader['CIEs'])
        WW[WW != 0] = 1
        C5=Connection(networkI, networkE, 'gi', weight=gIE*WW)
        WW = sparse.csr_matrix((loader['CIId'], loader['CIIi'], loader['CIIp']), shape=loader['CIIs'])
        WW[WW != 0] = 1
        C6=Connection(networkI, networkI, 'gi', weight=gII*WW)

    else:
        # generate a new random sparse connectivity with the prescribed structure
        fE = float(KE)/float(NE)
        fI = float(KI)/float(NI)
        C1=Connection(networkE, networkE, 'gea', weight=gEEA, sparseness=lambda i,j: conn(float(i)/NE-float(j)/NE,sigEE,fE))
        C2=Connection(networkE, networkE, 'gen', weight=gEEN/gEEA*C1.W)
        C3=Connection(networkE, networkI, 'gea', weight=gEIA, sparseness=lambda i,j: conn(float(i)/NE-float(j)/NI,sigEI,fE))
        C4=Connection(networkE, networkI, 'gen', weight=gEIN/gEIA*C3.W)
        C5=Connection(networkI, networkE, 'gi', weight=gIE, sparseness=lambda i,j: conn(float(i)/NI-float(j)/NE,sigIE,fI))
        C6=Connection(networkI, networkI, 'gi', weight=gII, sparseness=lambda i,j: conn(float(i)/NI-float(j)/NI,sigII,fI))

        # save the connectivity for future (faster) simulations with this network
        CEE = sparse.csr_matrix(C1.W)
        CEI = sparse.csr_matrix(C3.W)
        CIE = sparse.csr_matrix(C5.W)
        CII = sparse.csr_matrix(C6.W)
        np.savez_compressed('connections_sp', CEEd=CEE.data, CEEi=CEE.indices, CEEp=CEE.indptr, CEEs=CEE.shape,
                                             CEId=CEI.data, CEIi=CEI.indices, CEIp=CEI.indptr, CEIs=CEI.shape,
                                             CIEd=CIE.data, CIEi=CIE.indices, CIEp=CIE.indptr, CIEs=CIE.shape,
                                             CIId=CII.data, CIIi=CII.indices, CIIp=CII.indptr, CIIs=CII.shape)

    # add short-term plasticity in the AMPA and NMDAR synapses between excitatory neurons
    mystpEEA=STP(C1,taud=taud,tauf=tauf,U=Ustp)
    mystpEEN=STP(C2,taud=taud,tauf=tauf,U=Ustp)

    # monitor the spikes of the network
    spikes=SpikeMonitor(networkE)

    # run the simulation before stimulus presentation
    run(stim_on,report='text')

    # define the stimulus input
    stimat = int(randint(0,nstims)/float(nstims)*NE) #stimulus location random within nstims possible equidistant locations
    pos=arange(NE)
    inpE=stimE*exp(-0.5*(pos/float(NE)-0.5)**2/(epsE**2))#stimE*(1.+epsE*cos(2*pi*(pos/float(NE)-0.5)))
    networkE.Iext= np.roll(inpE,stimat-NE/2)
    pos=arange(NI)
    inpI=stimI*(1.+epsI*cos(2*pi*(pos/float(NI)-0.5)))
    networkI.Iext= np.roll(inpI,int(stimat*NI/float(NE)-NE/2))

    # run the simulation during stimulus presentation
    run(stim_off-stim_on,report='text')

    # remove the stimulus input
    networkE.Iext=0*mV
    networkI.Iext=0*mV

    # monitor spike counts and input currents during delay
    counts=SpikeCounter(networkE)
    currRec=StateMonitor(networkE, 'Irec', record=True, when='start')

    # now run the rest of the simulation, the delay period
    run(runtime-stim_off,report='text')

    # convert delay spike counts to rates
    rates=counts.count/(runtime-stim_off)

    # get the rastergrams, neurons=i, time=t
    i,t         = spikes.it
    
    # obtain population rates from spike counts in 250ms windows at different time points into delay
    fr_delay = np.zeros(NE)
    fr_0sec  = np.zeros(NE)
    fr_1sec  = np.zeros(NE)
    fr_3sec  = np.zeros(NE)
    for n in range(NE):
        fr_delay[n] = 4*sum(i[t>=runtime-250*ms] == n)
        fr_0sec[n]  = 4*sum(i[(t>=stim_off) & (t<stim_off+250 * ms)] == n)
        fr_1sec[n]  = 4*sum(i[(t>=stim_off+750*ms) & (t<stim_off+1000*ms)] == n)
        fr_3sec[n]  = 4*sum(i[(t>=stim_off+2750*ms) & (t<stim_off+3000*ms)] == n)
    fr_delay = np.roll(fr_delay, NE/2-stimat)
    fr_0sec = np.roll(fr_0sec, NE/2-stimat)
    fr_1sec = np.roll(fr_1sec, NE/2-stimat)
    fr_3sec = np.roll(fr_3sec, NE/2-stimat)

    # get angle decoded from population activity at different time points into delay
    dec_0sec   = decode(fr_0sec, NE)
    dec_1sec   = decode(fr_1sec, NE)
    dec_3sec   = decode(fr_3sec, NE)
    
    # get the decoded angle through the whole simulation time
    popdectm, nwins = readout(i, t, runtime, NE)
    popdectm = np.array(popdectm)
    # now translate the decoded angle to the same stimulus location to be able to average over many trials with different stimuli
    popdectm = np.angle(np.exp(1j*(popdectm - stimat/float(NE)*2*np.pi))) + np.pi

    # append the results as a new line to output files that other parallel simulations share for output
    with open(beh_log, 'a') as myfile:
	myfile.write(log_file\
	    +"; "+str(stimat)+"; "+str(dec_0sec)+"; "+str(dec_1sec)+"; "+str(dec_3sec)\
	    +"; "+str(popdectm.tolist() )+'\n')

    with open(bmp_log, 'a') as myfile:
	myfile.write(log_file+'; '+str(stimat)+"; "+str(list(fr_1sec))+'; '+str(list(fr_delay))+'\n')

    from scipy.signal import detrend
    irec = currRec.values/mV
    irec_dt = detrend(irec)

    R=np.corrcoef(irec_dt)

    indE = range(NE)
    corrF = np.zeros(NE/2-1)
    for i in range(NE/2-1):
        corrF[i]=np.mean(R[indE,np.roll(indE,i+1)])

    irec_sd = np.nanstd(irec_dt,axis=1)
    irec_mn = np.nanmean(irec, axis=1)

    with open(irec_log, 'a') as myfile:
        myfile.write(log_file+'; '+str(stimat)+'; '+str(corrF.tolist() )+"; "+str(irec_sd.tolist() )+"; "+str(irec_mn.tolist() )+'\n')


    print 'saved'

#####################################################################################################
#                                    RUN SIMULATIONS                                                #
#####################################################################################################

# run as many simulations as cores are available. Each simulation will append results to the same output file
pool = mp.Pool(processes=numcores, maxtasksperchild=1)   
myseeds = range(0, numcores)
args = [(sd) for sd in myseeds]
pool.map(run_simulation, args, chunksize=1)
#

print 'all sims finished'
print time.time() - start_time 

