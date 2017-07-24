##
## Copyright (C) 2017, Amit Aides, all rights reserved.
## 
## This file is part of Camera Network
## (see https://bitbucket.org/amitibo/cameranetwork_git).
## 
## Redistribution and use in source and binary forms, with or without modification,
## are permitted provided that the following conditions are met:
## 
## 1)  The software is provided under the terms of this license strictly for
##     academic, non-commercial, not-for-profit purposes.
## 2)  Redistributions of source code must retain the above copyright notice, this
##     list of conditions (license) and the following disclaimer.
## 3)  Redistributions in binary form must reproduce the above copyright notice,
##     this list of conditions (license) and the following disclaimer in the
##     documentation and/or other materials provided with the distribution.
## 4)  The name of the author may not be used to endorse or promote products derived
##     from this software without specific prior written permission.
## 5)  As this software depends on other libraries, the user must adhere to and keep
##     in place any licensing terms of those libraries.
## 6)  Any publications arising from the use of this software, including but not
##     limited to academic journal and conference publications, technical reports and
##     manuals, must cite the following works:
##     Dmitry Veikherman, Amit Aides, Yoav Y. Schechner and Aviad Levis, "Clouds in The Cloud" Proc. ACCV, pp. 659-674 (2014).
## 
## THIS SOFTWARE IS PROVIDED BY THE AUTHOR "AS IS" AND ANY EXPRESS OR IMPLIED
## WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
## MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
## EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
## INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
## BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
## DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
## LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
## OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
## ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.##
from __future__ import division
import cPickle
import datetime
from dateutil import parser
import itertools
import math
import numpy as np
import os
import pandas as pd
from scipy.integrate import quadrature, romberg, quad, fixed_quad
from scipy.interpolate import interp1d
from scipy.special import legendre
from tqdm import tqdm


TIMES = [pd.datetime(2017, 4, 22, 7, 30, 0) + i * datetime.timedelta(0, 3600) for i in range(9)]
WAVELENGTH = 532


def dateparse(d, t):
    return pd.datetime.strptime(d+' '+t, '%d:%m:%Y %H:%M:%S')


def parsePhaseFunction(path, usecols):
    """Parse the sunphotometer Phase Function data."""

    df = pd.read_csv(
        path,
        skiprows=3,
        header=0,
        parse_dates=[[0, 1]],
        date_parser=dateparse,
        index_col=0,
        usecols=usecols
    )

    wavelengths = [float(i.split("[")[0]) for i in df.columns]
    df = df.rename(columns=dict(itertools.izip(df.columns, wavelengths)))
    df.index.rename("datetime", inplace=True)
    
    for t in TIMES:
        new_row = pd.Series(name=t, data=[None]*len(df.columns), index=df.columns)
        df = df.append(new_row)
    df = df.sort_index().interpolate(method="pchip")

    return df, wavelengths


def interpolate_phasefunction(pfn_path, t):
    df_441, wavelengths = parsePhaseFunction(
        pfn_path,
        usecols=[0, 1]+list(range(3, 86))
    )
    df_674, wavelengths = parsePhaseFunction(
        pfn_path,
        usecols=[0, 1]+list(range(86, 169))
    )
    df_871, wavelengths = parsePhaseFunction(
        pfn_path,
        usecols=[0, 1]+list(range(169, 252))
    )
    df_1020, wavelengths = parsePhaseFunction(
        pfn_path,
        usecols=[0, 1]+list(range(252, 335))
    )
    
    series = [pd.Series(d.loc[t], name=n) for d, n in zip((df_441, df_674, df_871, df_1020), (441., 674., 871., 1020.))]
    df_phf = pd.concat(series, axis=1).T
    new_row = pd.Series(name=WAVELENGTH, data=[None]*len(df_phf.columns), index=df_phf.columns)
    df_phf = df_phf.append(new_row).sort_index().interpolate(method="pchip")
    
    return df_phf, wavelengths


def interpolate_SSA(path, t):
    """Parse the sunphotometer SSA data."""

    df = pd.read_csv(
        path,
        skiprows=3,
        header=0,
        parse_dates=[[0, 1]],
        date_parser=dateparse,
        index_col=0,
        usecols=[0, 1]+list(range(3, 7))
    )

    wavelengths = [float(i[3:-2]) for i in df.columns]
    df = df.rename(columns=dict(itertools.izip(df.columns, wavelengths)))
    df.index.rename("datetime", inplace=True)
    
    new_row = pd.Series(name=t, data=[None]*len(df.columns), index=df.columns)
    df = df.append(new_row)
    df = df.sort_index().interpolate(method="pchip")

    new_row = pd.Series(name=WAVELENGTH, data=[None]*len(df.T.columns), index=df.T.columns)
    df = df.T.append(new_row).sort_index().interpolate(method="pchip")
    
    return df


def calculate_legendre(df_phf, wavelengths, COFFES_NUM=300):

    mu = np.cos(np.radians(wavelengths[::-1]))
    phase = df_phf.loc[WAVELENGTH].values[::-1]
    
    phase_interp = interp1d(mu, phase, kind=1)
    
    def func(x, l, lg):
        return (2*l + 1)/2. * lg(x)*phase_interp(x)
    
    coeffs_quad = []
    for l in tqdm(range(COFFES_NUM)):
        lg = legendre(l)
        #coeffs_quad.append(quad(func, -1, 1, args=(l, lg,), limit=1000)[0])
        #coeffs_quad.append(fixed_quad(func, -1, 1, args=(l, lg,), n=50000)[0])
        coeffs_quad.append(quadrature(func, -1, 1, args=(l, lg,), maxiter=1000)[0])
    
    coeffs_quad[0] = 1.
    
    return coeffs_quad


def save_table(dst_path, df_ssa, coeffs_quad, t):
    
    header = """! Mie scattering table vs. effective radius (LWC=1 g/m^3)
      0.{}    0.{}    0.000  wavelength range and averaging step (micron)
       2    1.450    1.450  number       starting       ending REAL refractive index
       2    0.000    0.000  number       starting       ending IMAGINARY refractive index
       2    0.100    0.100  number       starting       ending effective radius
       2    0.500    0.500  number       starting       ending effective variance
     """.format(WAVELENGTH, WAVELENGTH)
    line_template = " 1.450 -.000     1.000    %f   0.5000   0.1000     %d  Phase function: Re{m}  Im{m}   Ext  Alb  Veff  Reff  Nleg\n"
    
    with open(dst_path, "w") as f:
        f.write(header)
        for i in range(700):
            f.write(line_template % (df_ssa.loc[WAVELENGTH][t], len(coeffs_quad)-1))
            f.write("\t".join(['']+[str(c) for c in coeffs_quad]))
            f.write("\n")
            

def main():
    pfn_path = r"../data/phase_functions/170422_170422_Technion_Haifa_IL.pfn"
    ssa_path = r"../data/phase_functions/170422_170422_Technion_Haifa_IL.ssa"

    for t in TIMES:
        df_phf, wavelengths = interpolate_phasefunction(pfn_path, t)
        
        coeffs_quad = calculate_legendre(df_phf, wavelengths)
        
        df_ssa = interpolate_SSA(ssa_path, t)
    
        dst_path = "aerosol{}dredvedm_{}.scat".format(WAVELENGTH, t.strftime("_%y%m%d_%H%M"))
        save_table(dst_path, df_ssa, coeffs_quad, t)


if __name__ == "__main__":
    main()