# VermeerKAT pipeline

(C) SARAO, Radio Astronomy Research Group

(C) Benjamin Hugo

![](https://upload.wikimedia.org/wikipedia/commons/0/0e/Johannes_Vermeer_-_The_Astronomer_-_WGA24685.jpg)
<p align="center">
 <i> The Astronomer, by Vermeer </i>
</p>

This is the temporary home of the VermeerKAT pipeline. The goal of this project is two-fold
 - to combine the various transfer, polarization and self calibration pipelines that I have laying around and implemented in the SKA-SA fleeting pol pipeline.
 - to implement a semi-interactive pipelining framework - something that is critically lacking from MeerKATHI

Currently the pipeline does flagging, transfer calibration and polarization calibration. The various tasks are available through the ```vermeerkat``` wrapper. Use ```vermeerkat --help```

# Functionality
The Basic Apply Transfer (BAT) pipeline is incorporated and feature complete. The pipeline uses the RARG tricolour flagger in combination with CASA and WSClean to perform transfer calibration (1GC). BAT can be invoked through:
```
vermeerkat transfer --help
```

Fleetingpol is now part of the Vermeerkat suite. You can run this standalone on any bandpass corrected dataset (like that produced from BAT which has transferred the solutions to 3C286 / 3C138. I recommend that you mark your crosshand phase calibrator as an ALTCAL in BAT to calibrate keep the system phased on the crosshand calibrator. See available options from 
```
vermeerkat poltransfer --help
```

# Installation
You need to have casacore-data installed on your system. 
The package should be installed via pip, preferably into a virtual environment
