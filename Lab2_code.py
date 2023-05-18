import math as m
import numpy as np
from matplotlib import pyplot as plt
from scipy import special
from scipy.optimize import curve_fit


def readfile():
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


def F(x, A, s, N, tau, alpha, x0, gamma):
    return A * (((1 - s) * fb(x, N, tau)) + s * V(x, x0, alpha, gamma))


def fb(x, N, tau):
    return N * np.exp((-x) / tau)


def V(x, x0, alpha, gamma):
    """
    Return the Voigt line shape at x with Lorentzian component HWHM gamma and Gaussian component HWHM alpha.
    """
    sigma = alpha / np.sqrt(2 * np.log(2))
    return (
        np.real(special.wofz(((x - x0) + 1j * gamma) / sigma / np.sqrt(2)))
        / sigma
        / np.sqrt(2 * np.pi)
    )


if __name__ == "__main__":
    # read and calculate data
    output = readfile()

    # plot histogram
    mass = [output[i][0] for i in range(len(output))]
    plt.figure(1)
    plt.style.use("ggplot")
    n, bin, _ = plt.hist(
        mass, bins=100, range=[70, 110], edgecolor="black", color="green"
    )
    plt.title(f"Invariant Mass of Muon Pairs", fontsize=10)
    plt.xlabel("Mass[GeV]", fontsize=8)
    plt.ylabel("Entries/bin", fontsize=8)
    # curve fitting
    bin_centers = (bin[:-1] + bin[1:]) / 2
    xdata = bin_centers
    ydata = n
    # [A,s,N,tau,alpha,x0,gamma]
    guess = [4000, 0.78, 30, 10, 2.7, 90, 1]
    popt, pcov = curve_fit(
        F,
        xdata,
        ydata,
        p0=guess,
        bounds=(
            [0, 0, 0, 0, 0, 0, -np.inf],
            [np.inf, 1, np.inf, np.inf, np.inf, np.inf, np.inf],
        ),
    )

    Fdata = [F(x, *popt) for x in xdata]
    plt.plot(xdata, Fdata, "r-")

    # use the calculated s from popt to plot background
    p_backgr = popt.copy()
    p_backgr[1] = 0
    print(p_backgr)
    Fdata1 = [F(x, *p_backgr) for x in xdata]
    plt.plot(xdata, Fdata1, "b-")

    plt.show()
    print(popt)
    print(pcov)
