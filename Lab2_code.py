import math as m
import numpy as np
from matplotlib import pyplot as plt
from scipy import special
from scipy.optimize import curve_fit
from scipy import integrate as integrate


def readfile(max_iso):
    """This function reads the mini_muons.txt file and computes the invariant mass of each particle collision."""
    directory = "/Users/shinnishida/Downloads/P121W research/Data/mini_muons.txt"
    output = []
    with open(directory, "r") as f:
        while True:
            line = f.readline()
            if "NumMuons: 2" in line:
                # ['Muon', '1', 'Pt', '13.122', 'Eta', '0.524', 'Phi', '-2.795', 'Charge', '-1', 'Iso', '1.277']
                muon1 = f.readline().split()
                muon2 = f.readline().split()
                # filter same charges
                if muon1[9] == muon2[9]:
                    continue
                # isolation requirement < max_iso then calculate invariant mass and store into output
                if (
                    0 <= float(muon1[11]) <= max_iso
                    and 0 <= float(muon2[11]) <= max_iso
                ):
                    # interested in indexes 3,5,7,9,11
                    px1 = float(muon1[3]) * m.cos(float(muon1[7]))
                    px2 = float(muon2[3]) * m.cos(float(muon2[7]))
                    py1 = float(muon1[3]) * m.sin(float(muon1[7]))
                    py2 = float(muon2[3]) * m.sin(float(muon2[7]))
                    pz1 = float(muon1[3]) * m.sinh(float(muon1[5]))
                    pz2 = float(muon2[3]) * m.sinh(float(muon2[5]))
                    muon_mass = 0.1057  # rest mass of muon in GeV
                    # Energy of each muon
                    E1 = m.sqrt(muon_mass**2 + px1 * px1 + py1 * py1 + pz1 * pz1)
                    E2 = m.sqrt(muon_mass**2 + px2 * px2 + py2 * py2 + pz2 * pz2)
                    invariant = m.sqrt(
                        2 * muon_mass**2
                        + 2 * E1 * E2
                        - 2 * (px1 * px2 + py1 * py2 + pz1 * pz2)
                    )
                    muon_dict = dict(
                        [
                            ("isolation", [float(muon1[11]), float(muon2[11])]),
                            ("charge", [float(muon1[9]), float(muon2[9])]),
                        ]
                    )
                    output.append((invariant, muon_dict))
            if line == "":
                break
    # print(len(output))
    return output


def histogram(data, bins, range, edgecolor, color, alpha, label):
    n, bin, _ = plt.hist(
        data,
        bins=bins,
        range=range,
        edgecolor=edgecolor,
        color=color,
        alpha=alpha,
        label=label,
    )
    return n, bin


def F(x, A, s, tau, x0, sigma, gamma):
    def integrand(x, x0, tau):
        return np.exp((-(x - x0)) / tau)

    def fb(x, x0, tau):
        integral, _ = integrate.quad(integrand, 70, 110, args=(tau, x0))
        return np.sqrt((1 / integral)) * np.exp((-(x - x0)) / tau)

    def V(x, x0, sigma, gamma):
        # sigma = alpha / np.sqrt(2 * np.log(2))
        return (
            np.real(special.wofz(((x - x0) + 1j * gamma) / sigma / np.sqrt(2)))
            / sigma
            / np.sqrt(2 * np.pi)
        )

    return A * (((1 - s) * fb(x, x0, tau)) + s * V(x, x0, sigma, gamma))


def FB(x, A, s, tau, x0):
    def integrand(x, x0, tau):
        return np.exp((-(x - x0)) / tau)

    def fb(x, x0, tau):
        integral, _ = integrate.quad(integrand, 70, 110, args=(tau, x0))
        return np.sqrt((1 / integral)) * np.exp((-(x - x0)) / tau)

    return A * ((1 - s) * fb(x, x0, tau))


def isolation_distribution():
    """this function plots the spread of isolation in each muon data"""
    output = readfile(10)
    iso1 = []
    iso2 = []
    for tuples in output:
        iso = tuples[1]["isolation"]
        iso1.append(iso[0])
        iso2.append(iso[1])
    plt.scatter(iso1, iso2, s=0.01, alpha=0.75)  # s: marker size, alpha: transparency
    return


def optimize(upper, lower, step):
    """this function optimizes the uncertainty using the curve_fit function using different values of isolation
    thresholds
    returns the isolation value that gives the minimizes uncertainty
    returns: [(isolation, signal frac, uncertainty),]"""
    steps = np.arange(upper, lower, step)
    iso_info = []
    output = readfile(10)
    for iso in steps:
        mass = [
            output[i][0]
            for i in range(len(output))
            if (
                output[i][1]["isolation"][0] <= iso
                and output[i][1]["isolation"][1] <= iso
            )
        ]
        n, bin = histogram(mass, 200, [70, 110], "black", "green", 0.75, "OS muons")
        # curve fit
        xdata = (bin[:-1] + bin[1:]) / 2
        ydata = n
        guess = [14000, 0.9, 50, 91.2, 1, 1]
        popt, pcov = curve_fit(
            F,
            xdata,
            ydata,
            p0=guess,
            bounds=(
                [0, 0, 0, 70, 0, 0],
                [100000, 1, 1000, 110, 10, 10],
            ),
        )
        sig_frac = popt[1]
        uncertainty = np.sqrt(pcov[1][1])
        iso_info.append((iso, sig_frac, uncertainty))
    return iso_info


if __name__ == "__main__":
    # isolation plot
    # plt.figure(1)
    # isolation_distribution()
    # plt.title(f"Spread of Muon Isolations", fontsize=10)
    # plt.xlabel("Isolation 1", fontsize=8)
    # plt.ylabel("Isolation 2", fontsize=8)
    # plt.show()

    # testing for one specific threshold (before implementing an algorithm)
    # read and calculate data
    output = readfile(0.157)
    # plot histogram
    mass = [output[i][0] for i in range(len(output))]
    plt.figure(2)
    plt.style.use("ggplot")
    n, bin = histogram(mass, 200, [70, 110], "black", "green", 0.75, "OS muons")
    # curve fitting
    bin_centers = (bin[:-1] + bin[1:]) / 2
    xdata = bin_centers
    ydata = n
    # [A,s,tau,x0,sigma,gamma]
    guess = [14000, 0.9, 50, 91.2, 1, 1]
    popt, pcov = curve_fit(
        F,
        xdata,
        ydata,
        p0=guess,
        bounds=(
            [0, 0, 0, 70, 0, 0],
            [100000, 1, 1000, 110, 10, 10],
        ),
    )

    Fdata = [F(x, *popt) for x in xdata]
    plt.plot(xdata, Fdata, "r--", label="Sig + Bg")

    # use the calculated s from popt to plot background
    p_backgr = popt.copy()[:4]
    Fdata1 = [FB(x, *p_backgr) for x in xdata]
    plt.plot(xdata, Fdata1, "b--", label="Background")

    # finish up figure
    plt.title(f"Invariant Mass of Muon Pairs", fontsize=10)
    plt.xlabel("Mass[GeV]", fontsize=8)
    plt.ylabel("Entries/bin", fontsize=8)
    plt.legend(loc="best", fontsize=10)
    plt.show()

    print("curve_fit parameters [A,s,tau,x0,sigma,gamma]")
    print(popt)
    print()
    # calculate uncertainty
    for i in range(len(pcov)):
        print(np.sqrt(pcov[i][i]))
    cov = pcov[1][1]
    print(f"Signal fraction: {popt[1]}\nUncertainty (St. dev):{np.sqrt(cov)}")

    # implement optimization algorithm and plot the histogram with the least uncertainty
    # iso_info = optimize(5, 0, -0.1)
    # optimal = [
    #     iso_info[i]
    #     for i in range(len(iso_info))
    #     if iso_info[i][2] == min([iso_info[j][2] for j in range(len(iso_info))])
    # ]
    # print(optimal)  # ans = 0.2

    # second scan
    # iso_info2 = optimize(0.3, 0.1, -0.01)
    # optimal2 = [
    #     iso_info2[i]
    #     for i in range(len(iso_info2))
    #     if iso_info2[i][2] == min([iso_info2[j][2] for j in range(len(iso_info2))])
    # ]
    # print(optimal2)  # ans = 0.16

    # # # third scan
    # iso_info3 = optimize(0.17, 0.15, -0.001)  # ans = 0.157
    # optimal3 = [
    #     iso_info3[i]
    #     for i in range(len(iso_info3))
    #     if iso_info3[i][2] == min([iso_info3[j][2] for j in range(len(iso_info3))])
    # ]
    # print(optimal3)  # ans = 0.157

    # plt.figure(3).tight_layout()
    # plt.subplot(1, 2, 1)
    # isos = [iso_info2[i][0] for i in range(len(iso_info2))]
    # sigf = [iso_info2[i][1] for i in range(len(iso_info2))]
    # unc = [iso_info2[i][2] for i in range(len(iso_info2))]
    # plt.plot(isos, unc, "o-")
    # plt.ylim([0.001, 0.003])
    # plt.title(f"Isolation vs Uncertainty of Signal Fraction", fontsize=10)
    # plt.xlabel("Isolation Threshold", fontsize=8)
    # plt.ylabel("Uncertainty", fontsize=8)

    # plt.subplot(1, 2, 2)
    # isos2 = [iso_info3[i][0] for i in range(len(iso_info3))]
    # sigf2 = [iso_info3[i][1] for i in range(len(iso_info3))]
    # unc2 = [iso_info3[i][2] for i in range(len(iso_info3))]
    # plt.plot(isos2, unc2, "o-")
    # plt.ylim([0.0018, 0.0019])
    # plt.title(f"Isolation vs Uncertainty of Signal Fraction", fontsize=10)
    # plt.xlabel("Isolation Threshold", fontsize=8)
    # plt.ylabel("Uncertainty", fontsize=8)
    # plt.show()
