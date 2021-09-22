import principal
import os

if not os.path.isdir("figures"):
    os.mkdir("figures")

np = principal.np
plt = principal.plt
ns = range(20,51,2)
m = 5
ts = 500
lifeFunction = principal.linearLife
vals = np.zeros((len(ns),m))
for i,k in enumerate(ns):
    radius = 0.05*np.ones((k,ts))
    for l in range(m):
        print(f"Advance: {(i*m+l)*100/(m*len(ns))} %")
        position, obs_pos, obs_visits, radius = principal.simulation(n_gel=k,t_steps=ts,radius=radius)
        vals[i,l] = np.mean([lifeFunction(x) for x in obs_visits[-1,:]])
fig = plt.figure()
plt.errorbar(ns,np.mean(vals,axis=1),yerr = np.std(vals,axis=1),fmt="o",ecolor="black",capsize=4)
plt.xlabel("número de caminantes")
plt.ylabel("vida media final ")
plt.grid()
plt.savefig(os.path.join("figures","n_gel.pdf"))
plt.close()


rs = np.linspace(0.01,0.1,21)
k = 40
vals = np.zeros((len(rs),m))
for i,r in enumerate(rs):
    radius = r*np.ones((k,ts))
    for l in range(m):
        print(f"Advance: {(i*m+l)*100/(m*len(rs))} %")
        position, obs_pos, obs_visits, radius = principal.simulation(n_gel=k,t_steps=ts,radius=radius)
        vals[i,l] = np.mean([lifeFunction(x) for x in obs_visits[-1,:]])
fig = plt.figure()
plt.errorbar(rs,np.mean(vals,axis=1),yerr = np.std(vals,axis=1),fmt="o",ecolor="black",capsize=4)
plt.xlabel("Radio")
plt.ylabel("vida media final ")
plt.titple("40 Caminantes")
plt.grid()
plt.savefig(os.path.join("figures","r.pdf"))
plt.close()