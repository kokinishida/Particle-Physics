import os
import math as m
import numpy as np
from matplotlib import pyplot as plt


def invariant_mass(name: str):
    """name: specify a string to filter the file name from signal to background
    returns: [] a list of all the invariant masses"""
    # read signal files
    directory = "/Users/shinnishida/Downloads/P121W research/Data/p121_lab1_data"
    mass_ = []
    for filename in os.listdir(directory):
        if name in filename:
            filepath = os.path.join(directory, filename)
            with open(filepath, "r") as f:
                while True:
                    line = f.readline()
                    if "NumElectrons: 2" in line:
                        # ['Electron', '1', 'Pt', '83.561', 'Eta', '-0.708', 'Phi', '-3.002', 'Charge', '-1']
                        # We are interested in indexes 3,5,7 which are pt,eta,phi
                        indexes = [3, 5, 7]
                        e1 = f.readline().split()
                        e2 = f.readline().split()
                        e1 = [float(e1[x]) for x in indexes]
                        e2 = [float(e2[x]) for x in indexes]
                        # convert to cartesian coords
                        px1 = e1[0] * m.cos(e1[2])
                        px2 = e2[0] * m.cos(e2[2])
                        py1 = e1[0] * m.sin(e1[2])
                        py2 = e2[0] * m.sin(e2[2])
                        pz1 = e1[0] * m.sinh(e1[1])
                        pz2 = e2[0] * m.sinh(e2[1])
                        mass_e = 0.000511  # rest mass of electron in GeV
                        # Energy of each electron
                        ene1 = m.sqrt(mass_e**2 + px1 * px1 + py1 * py1 + pz1 * pz1)
                        ene2 = m.sqrt(mass_e**2 + px2 * px2 + py2 * py2 + pz2 * pz2)
                        invariant = m.sqrt(
                            2 * mass_e**2
                            + 2 * ene1 * ene2
                            - 2 * (px1 * px2 + py1 * py2 + pz1 * pz2)
                        )
                        mass_.append(invariant)
                    # end of the txt file
                    if line == "":
                        break
    # print(len(mass_))
    return mass_


def plot_hist(
    s_mass: str, b_mass: str, bins: int, window: list, s_weight: float, b_weight: float
):
    """Plots a histogram of background data & background+signal data
    Inputs:
        s_mass,b_mass: str to determine which file to read
        bins: # of bins
        weight: number to weight
        ith: is the ith graph in the figure
        returns: the significance N_s/sqrt(N_b)
    """
    signal_mass = invariant_mass(s_mass)
    bg_mass = invariant_mass(b_mass)
    combined_mass = signal_mass + bg_mass
    weight_bg = list(np.full(len(bg_mass), b_weight))
    weight_signal = list(np.full(len(signal_mass), s_weight))
    weight_combined = list(np.full(len(signal_mass), s_weight)) + list(
        np.full(len(bg_mass), b_weight)
    )

    # plt.style.use('ggplot')

    # plot SIGNAL
    n_s, bin_s, _ = plt.hist(
        combined_mass,
        bins=bins,
        range=window,
        edgecolor="black",
        label="Signal+Background",
        alpha=0.75,
        weights=weight_combined,
    )
    mode_index = n_s.argmax()
    peak = (bin_s[mode_index + 1] + bin_s[mode_index]) / 2
    bin_width = bin_s[mode_index + 1] - bin_s[mode_index]

    # plot BACKGROUND
    n_b, bin_b, _ = plt.hist(
        bg_mass,
        bins=bins,
        range=window,
        edgecolor="black",
        label="Background",
        alpha=0.75,
        weights=weight_bg,
    )

    # plot signal only
    n, bin, _ = plt.hist(
        signal_mass,
        bins=bins,
        range=window,
        edgecolor="black",
        label="Signal",
        alpha=0,
        weights=weight_signal,
    )  # hidden on histogram
    plt.title(f"# of Events vs Mass [{s_mass}GeV]", fontsize=10)
    plt.xlabel("Mass[GeV]", fontsize=8)
    plt.ylabel("Entries/bin", fontsize=8)
    plt.legend(loc="best", fontsize=7)

    # optimize window
    optimize = []
    delta = bin_width / 2  # we only take the first bin first and compute significance
    Nsignal = n[mode_index]
    Nbg = n_b[mode_index]
    significance = Nsignal / m.sqrt(Nbg)
    optimize.append((peak, delta, significance))
    # now we run a for loop to increment delta with the bin_width
    for i in range(1, int((bins - 1) / 2)):
        delta = delta + bin_width
        Nsignal = sum(n[mode_index - i : mode_index + i + 1])
        Nbg = sum(n_b[mode_index - i : mode_index + i + 1])
        significance = Nsignal / m.sqrt(Nbg)
        optimize.append((peak, delta, significance))

    return optimize


if __name__ == "__main__":
    # [signal,bg,bin,range,weights]
    h_input = [
        ["200", "175", 40, [150, 250], 3, 2],
        ["300", "275", 40, [250, 350], 2, 2],
        ["400", "375", 40, [350, 450], 1.5, 2],
        ["500", "450", 40, [450, 550], 1, 2],
        ["750", "650", 40, [700, 800], 0.75, 2],
        ["1000", "900", 40, [950, 1050], 0.5, 2],
    ]

    # make subplot
    plt.figure(1, figsize=(11, 9)).tight_layout()
    plt.style.use("ggplot")
    peaks = []
    deltas = []
    sigs = []

    for i in range(len(h_input)):
        plt.figure(1)
        plt.style.use("ggplot")
        plt.subplot(2, 3, i + 1)
        a = plot_hist(
            h_input[i][0],
            h_input[i][1],
            h_input[i][2],
            h_input[i][3],
            h_input[i][4],
            h_input[i][5],
        )
        optimal = [
            a[y]
            for y in range(len(a))
            if a[y][2] == max([a[x][2] for x in range(len(a))])
        ]
        print(f"[{h_input[i][0]}GeV] (peak,delta,significance)")
        print(a)
        print(
            f"From the calculations, the significance is maximized at {optimal[0][2]} when delta is {optimal[0][1]} with the peak at {optimal[0][0]}"
        )
        peaks.append(optimal[0][0])
        deltas.append(optimal[0][1])
        sigs.append(round(optimal[0][2], 5))
        print()

        # delta vs significance
        x = []
        y = []
        for j in range(len(a)):
            x.append(a[j][1])
            y.append(a[j][2])
        plt.figure(2)
        plt.style.use("tableau-colorblind10")
        plt.plot(x, y, label=f"{h_input[i][0]} GeV", marker=".")
        plt.xlabel("Delta", fontsize=8)
        plt.ylabel("Significance", fontsize=8)
        plt.legend(loc="best", fontsize=7)
    plt.show()

    # table
    plt.style.use("tableau-colorblind10")
    plt.figure(3, figsize=(11, 11))
    plt.axis("off")
    labels = ["Mass of Z' (GeV)", "Optimal Delta", "Significance"]
    data = [peaks, deltas, sigs]
    print(data)
    table = plt.table(
        cellText=data,
        cellLoc="center",
        colWidths=[0.1] * 6,
        rowLabels=labels,
        loc="center",
    )
    table.set_zorder(100)

    plt.show()


# try reading a single file

# with open('/Users/shinnishida/Downloads/P121W research/Data/p121_lab1_data/zp_mzp750_electrons.txt','r') as f:
#     mass_list = []
#     while True:
#         line = f.readline()
#         if 'NumElectrons: 2' in line:
#             # ['Electron', '1', 'Pt', '83.561', 'Eta', '-0.708', 'Phi', '-3.002', 'Charge', '-1']
#             # We are interested in indexes 3,5,7 which are pt,eta,phi
#             indexes = [3,5,7]
#             e1 = f.readline().split()
#             e2 = f.readline().split()
#             e1 = [float(e1[x]) for x in indexes]
#             e2 = [float(e2[x]) for x in indexes]
#             mass_e = 0.000511 # in GeV
#             ene1 = math.sqrt(mass_e**2 + sum([x**2 for x in e1]))
#             ene2 = math.sqrt(mass_e**2 + sum([x**2 for x in e2]))
#             invariant = math.sqrt(2*mass_e**2 + 2*ene1*ene2 - 2*sum([e1[x]*e2[x] for x in range(3)]))
#             mass_list.append(invariant)
#         # end of the txt file
#         if line == "":
#             break
#     print(mass_list)
#     print(len(mass_list))
