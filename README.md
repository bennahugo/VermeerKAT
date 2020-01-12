# VermeerKAT pipeline

(C) SARAO, Radio Astronomy Research Group

(C) Benjamin Hugo

![](https://upload.wikimedia.org/wikipedia/commons/0/0e/Johannes_Vermeer_-_The_Astronomer_-_WGA24685.jpg)
<p align="center">
 <i> The Astronomer, by Vermeer </i>
</p>

This is the home of the VermeerKAT pipeline. The goal of this project is two-fold
 - to combine the various transfer, polarization and self calibration pipelines that I have laying around and implemented in the SKA-SA fleeting pol pipeline.
 - to implement a semi-interactive pipelining framework

Currently the pipeline does flagging, transfer calibration, polarization calibration and 2GC self-calibration. The various tasks are available through the ```vermeerkat``` wrapper. Use ```vermeerkat --help```

# Functionality
The Basic Apply Transfer (BAT) pipeline is feature complete. The pipeline uses the RARG tricolour flagger in combination with CASA and WSClean to perform transfer calibration (1GC). BAT can be invoked through:
```
vermeerkat transfer --help
```

Fleetingpol is now part of the Vermeerkat suite. You can run this standalone on any bandpass corrected dataset (like that produced from BAT which has transferred the solutions to 3C286 / 3C138), with the caveat that the corrected calibrator data is all stored in CORRECTED_DATA). I recommend that you mark your crosshand phase calibrator as an ALTCAL in BAT to calibrate keep the system phased on the crosshand calibrator. See available options from 
```
vermeerkat poltransfer --help
```

**NOTE:**
A known issue with MeerKAT data is that both Q and V is flipped in sign with respect to the IEEE convention. After calibration it is necessary to flip signs in analysis. The fleetingpol pipeline forms a SKY_CORRECTED_DATA column which can be used for imaging in the derotated sky frame. Any further self calibration should be performed with the CORRECTED_DATA column. Ideally this pipeline is run post self-calibration of the phase of the target field.

The Introspect self-calibration pipeline is a configurable self-calibration pipeline with capacity to apply delay, phase and amplitude self-calibration using CASA and WSClean. A recipe can be specified as follows:
```
p(35,256s),p(25,64s),dp(15,16s),ap(7,16s),i(CORRECTED_DATA,0.0),s,i(SUBTRACTED_DATA,0.0)
```

- Available options for cal are p - phase, dp - delay+phase, ap - ampphase.
- Available options for im are currently only i with customization of image column and robust weighting.
- s subtracts MODEL_DATA from CORRECTED_DATA to form SUBTRACTED_DATA for all fields.

Full help is available by running
```
vermeerkat selfcal --help
```

# Installation
You need to have casacore-data installed on your system. 
The package should be installed via pip, preferably into a virtual environment
