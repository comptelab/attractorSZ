import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

files1 = glob.glob('output_beh*')

beh = pd.DataFrame()

for file in files1:
    df = pd.read_csv(file, delimiter=';', names=['log_file', 'stimat', 'readout0', 'readout1', 'readout3', 'popvec'])
    beh=beh.append(df, ignore_index=True)
    
files = glob.glob('output_bump*')

bmp = pd.DataFrame()

for file in files:
    df = pd.read_csv(file, delimiter=';', names=['log_file','stimat','bump1','bumpD'])
    bmp=bmp.append(df, ignore_index=True)

# plot delay activity firing rate
plt.subplot(1,4,1) 
beh['diff1']=np.abs(beh['readout1']-np.pi)
centeredtrials=beh['diff1'].sort_values()
bp=[]
for i in centeredtrials.index[:100]:
    bp.append(eval(bmp['bump1'][i]))
bp=np.array(bp)
xval = np.linspace(0,360,bp.shape[1])
yval=np.mean(bp,0)
plt.plot(xval,yval)
plt.xlabel('neuron (°)')
plt.ylabel('firing rate (sp/s)')


# plot mean multiplicative noise in delay
def get_noise(df):
    nsim = 1 #to average over multiple simulations, max = df.shape[0]
    cr=[]
    for i in range(nsim):
        mean_curr=np.array(eval(df['irec_mn'][i]))
        std_curr=np.array(eval(df['irec_sd'][i]))
        box_pts=50
        box = np.ones(box_pts)/box_pts
        std_curr=np.convolve(std_curr,box,mode='same')
        mean_curr=np.convolve(mean_curr,box,mode='same')
        cr.append(std_curr/mean_curr)
    
    return np.mean(cr,0)

plt.subplot(1,4,2) 
irec = pd.read_csv('output_irec1.txt', delimiter=';', names=['log', 'stim', 'corr', 'irec_sd', 'irec_mn'])
box_pts=100
box = np.ones(box_pts)/box_pts
lenx = len(eval(irec['irec_sd'][0]))
myx = np.linspace(0,360,lenx)
myx = myx[int(box_pts/2)-1:-int(box_pts/2)]
yval=np.convolve(get_noise(irec),box,mode='valid')
plt.plot(myx,yval)
plt.xlim([-10,370])
plt.xticks([0,180,360])
plt.xlabel('neuron (°)')
plt.ylabel('noise CV')

# plot spatial correlation of the noise in delay
def get_corr(df):
    nsim = 1 #to average over multiple simulations, max = df.shape[0]
    cr=[]
    for i in range(nsim):
        cr.append(eval(df['corr'][i]))
        
    return np.mean(cr,0)

box_pts=100
box = np.ones(box_pts)/box_pts
lenx = len(eval(irec['corr'][0]))
myx = np.linspace(0,180,lenx)
myx = myx[int(box_pts/2)-1:-int(box_pts/2)]
yval=np.convolve(get_corr(irec),box,mode='valid')
plt.plot(myx,yval)
plt.xlim([0,180])
plt.xticks([0,90,180])
plt.xlabel('distance (°)')
plt.ylabel('noise correlation')



# plot variance of population decoding through delay
plt.subplot(1,4,4)
bmp['meanact']=bmp.bumpD.apply(lambda x: np.std(eval(x)))
beh=beh[bmp['meanact']>np.median(bmp['meanact'])].reset_index()
pv=[]
for i in range(len(beh)):
    pv.append(eval(beh['popvec'][i]))
pv=np.array(pv)/np.pi*180
varpv = np.var(pv,axis=0)
xval = np.arange(varpv.shape[1])*0.1-2
plt.plot(xval, varpv)
plt.ylim([0,50])
plt.xlim([-0.5, 3])
plt.ylabel('decoder variance (deg²)')
plt.xlabel('time into delay (s)')


plt.tight_layout()

plt.show()
