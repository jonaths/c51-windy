import matplotlib.pyplot as plt
import numpy as np


def plot(states_labels, support, Qs):
    def softmax_rows(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.tile(np.max(x, axis=1), (x.shape[1], 1)).transpose())
        return np.divide(e_x, np.tile(e_x.sum(axis=1), (x.shape[1], 1)).transpose())

    tau = 0.5
    policy = np.array(
        [softmax_rows(Qs[i, :, :].transpose() / tau).transpose() for i in
         range(Qs.shape[0])])
    print(states_labels)
    fig = plt.figure(figsize=(4, 3))
    plt.imshow(policy[:, 1, :].transpose(), aspect='auto', cmap=plt.get_cmap('gray'),
               origin='lower')
    plt.xticks(np.arange(3), ['S' + str(s) for s in states_labels])
    plt.xlim([-0.5, 2.5])
    plt.xlabel("State")
    plt.yticks(
        np.linspace(0, policy.shape[2] - 1, 5),
        np.linspace(np.min(support), np.max(support), 5))
    plt.ylabel("Budget")
    plt.clim([0, 1])
    cbar = plt.colorbar(ticks=[0, 1])
    cbar.ax.set_yticklabels(['Left (risky)', 'Right (safe)'])
    # fig.savefig('samplefigure.pdf', bbox_extra_artists=(cbar.ax,),
    #             bbox_inches='tight')
    plt.show()