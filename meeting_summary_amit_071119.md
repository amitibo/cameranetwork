 -  DLL files  - core.so . Just keep what is working ! He used the core.so that was compiled in SIPL, in AWS. Recompilation - could be not trivial.
- Suggesting : use Mitzuba.
-  Suggesting to use AWS EC2 .
    -  Run SHDOM on optimized memory machines (at least 100 GB) .Check compute 2,3 or R .
    - Spot instance.
    - EFS for file storage (similar to dropbox) not S3. allows multi connection from several instances.
    - Ask for grant/discount.
    - Names of folders - YYMMDDHHmm(dateTime)_hash/code vesion/commit id


- GUI :

    - Map , TODO: ask Vadim where he got it . Amit had to do hand stitching.
    - 2 level of cropping : 1) ROI - rectangle to remove irrelevant  parts (e.g. if the camera is not seeing fully the reconstructed volume). 2) mask - circle to remove constant obstacles (buildings, trees) , note - do it once when installing/opening the camera . Saved under /data folder. 3) Grabcat - for sunshader (use high valuew e.g. 40-60) , can be seen under "mask" view .
    - grid change can be done in export settings.
    - use load and save of ROI . possible solution is to open *.pkl that were save and to convert them to a new version of pandas.
    - Space carving - the "dilate" is possibly extrude operation. TODO: Check it in the code.
    - score of space carving: the first is related (in 2D - inside "image settings"/"view settings" ) to images segmentation of cloud,
     the second (in 3D - inside inside "space carving" in the map view) is related to the
     probability that a voxel contains cloud. TODO: check it once in the GUI and second in the pyshdom usage.
     - Resolution  can be controoled in the main view or in view settings. TODO: check that it affects also on the masks, this is for doing upsampling later.
     - suggested to image one in 5 min in regular days, in good days every 1 minute.
     - radiometric calibration according to fitting with sunphotometer measurments (p.74-78 ). TODO: check this.
     - extrinsic calibration - according to sun position and sunshader, do it in a sunny day. (p.69-71). TODO: check this.
     - dropbox is legacy.
     - GUI intrinsic calibration is legacy (before the gimbal)
     - SSH: try to do ssh from camera to proxy once, and then it suppose to save the reverse tunnle properties.

- SHDOM: we went throught the: camera array retrival.py
    - TODO: check the calling to shdom to make sure what are the parameters. should be particles extinction coeeficient, dansity ... need to check
    pose and
    - mie table - is the SHDOM particle parameters.
    - beta is only water particles.
    - If the scripts is not for RGB, then it's only Red channles ( they assume the bluew is noiseir)
    - shdom is solving the volume seperated to the frequency/channels . and the differences between are the input each
    proccess is recoveing (e.g. cross section in blue or red ..)
    - It is better to debug procces id = 0 ( The main thread )
    - The RGB runs was using modulo . TODO: Assure the following : each proccess is runinng on a single shannel of a camera.
    e.g. if we have 10 cameras, and we want to solve in RGB that we get 30 proccess .
    - visibility.pkl - probably space carving . TODO - need to check .
    - TODO: learn mpi4py with relating to the scripts.
    - SHDOM doesn't care about units,and coordinate system - the input is controlled by us.
    - Coordinate system was hard and tough, what is working now - try to follow it.
    - modle.particles - in [km]
    - layered particles - this is probaly the lidar readings
    - subsample - to take several pixels measurments (reducing power of computation) , Amit used 1:2 or 1:4.
     TODO: debug to see if this is a flag or number.
     - TODO: understand the inverse model and camera model. the inverse model is inharent from "forward model".
     - When they included aerosols, that they didn't used space carving. They initiaited the aerosol accoring to lidar readings, and then solving the clouds accordingly.
     TODO: Ask Aviad about the mixture sollution he did . Aviad also wrote the gradient of shdom, and added an option to
     do optimization phisical propeties, and created the mie table.
     - The solution in each voxel is a linear combination of several solutions of the mie table.





Ask from amit to send us:
- Code that her saved on EFS
- Cameranetwork notebook
- check history on bitbucket.
