import pywed as pwd
import matplotlib.pyplot as plt
shot_nums = [57637,  57691, 58000]

# signal_name='SMAG_IP'
# signal_name = 'GINTLIDRT%3'
signal_names = ['SMAG_IP', 'GINTLIDRT%3']
plt.figure(dpi=150)
for shot_num in shot_nums:
    for signal_name in signal_names:
        ip, ip_t= pwd.tsbase(shot_num, signal_name, nargout=2)
        # print(ip)
        plt.plot(ip_t, ip)
plt.yscale('log')
plt.savefig('a.png')
