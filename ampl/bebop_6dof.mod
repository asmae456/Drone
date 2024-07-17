#####################################################################################
#
# Problem:  Optimal Control Problem (OCP)
# Dynamics: A quadcopter with two thrusters (right and left) as requested by Delft
# Transcription: Hermite-Simpson
#
# Author: Dario Izzo (Nov 2018)
#
#####################################################################################

#Parameters---------------------------------------------------
#Generic
    param n default          40;                  #Number of nodes
    param gearth:=           9.81;                #[m/s^2] Earth gravity constant
    param epsilon default    0.1;                 #Tunes the aggressivity of the optimal solution
    param pi:=               4*atan(1);

#Quadrotor params
    param mass default         0.389;                        # [kg] Mass
    param Ixx default          0.001242;                     # [kg m^2]
    param L default            0.08;                         # [m] Distance thrust center of mass
    param maxthrust default    2.35;                         # [N] Max thrust (per side)
    param minthrust default    1.76;                         # [N] Min thrust (per side)
    param beta default         0.5;                          # Drag coefficient (positive)
    param M default           (maxthrust - minthrust);       # [N] Maximum torque (for L=1.)
    param F default            2*minthrust;                  # [N] Maximum upward thrust

#State constraint params
    param maxtheta default pi/3.0;
    
#Initial conditions
    param y0 default        5.7;         #[m] Initial y
    param z0 default        3.9;         #[m] Initial z
    param vy0 default      -2.07;        #[m/s] Initial velocity in y
    param vz0 default       1.09;        #[m/s] Initial Velocity in z
    param theta0 default    0.67;        #[rad] Initial pitch
    param omega0 default    0.1;         #[rad/sec] Initial pitch rate

#Final conditions
    param yn default        0.0;         #[m] Final y
    param zn default        0.0;         #[m] Final z
    param vyn default       0.0;         #[m/s] Final velocity in y
    param vzn default       0.0;         #[m/s] Final velocity z
    param thetan default    0.0;         #[rad] Final pitch
    param omegan default    0.0;         #[rad/sec] Final pitch rate

#Other
    param tn:=1.;               #[s] Guess for the final time
#-------------------------------------------------------------

#Sets---------------------------------------------------------
    set I := 1..n;
    set J := 1..n-1;
#-------------------------------------------------------------

#Variables---------------------------------------------------
    var y{i in I};
    var vy{i in I};
    var z{i in I};
    var vz{i in I};
    var theta{i in I};
    var omega{i in I};
    var ul{i in I}, >=0, <=1;
    var ur{i in I}, >=0, <=1;
    var ulm{i in J}, >=0, <=1;
    var urm{i in J}, >=0, <=1;
#-------------------------------------------------------------

#Time variables-----------------------------------------------
    var tf, >=0;
    var dt = tf/(n-1);
    var timegrid{i in I} = dt*(i-1);
#-------------------------------------------------------------

#Objective----------------------------------------------------

        # For power, minimize Simpson's approximation to the integral:
        #
        #        \int{ f(t)dt }
        #     ~= \sum_{  dt/6 * f(t) + 4*f(t+dt/2)  + f(t+dt)  }
        #               for t=(dt,2*dt,3*dt...)
        #cost has the values at t = i*dt
        #costm has the values at t = i*dt + dt/2

    var cost{i in I} = ur[i]^2 + ul[i]^2;
    var costm{i in J} = urm[i]^2 + ulm[i]^2;
    #var cost{i in I} = log(ur[i] * ul[i] * (1-ur[i]) * (1-ul[i]));
    #var cost_m{i in J} = log(urm[i] * ulm[i] * (1-urm[i]) * (1-ulm[i]));
    var smoothing_term = dt/6 * sum{i in J} (cost[i]+4*costm[i]+cost[i+1]);

    minimize myobjective: smoothing_term * epsilon + tf * (1 - epsilon);

#-------------------------------------------------------------

#Dynamic at the grid points-----------------------------------
    var f1{i in I} = vy[i];
    var f2{i in I} = -((ur[i]+ul[i]) / mass * M + F / mass) * sin(theta[i]) - beta * vy[i];
    var f3{i in I} = vz[i];
    var f4{i in I} = ((ur[i]+ul[i]) / mass * M + F / mass) * cos(theta[i]) - gearth - beta * vz[i];
    var f5{i in I} = omega[i];
    var f6{i in I} = L / Ixx * M * (ur[i] - ul[i]);

#-----------------------------------------------------------------------

#State definition at mid-points via Hermite interpolation---------------
    var ym{i in J}      =   (    y[i] +     y[i+1])/2 + tf/(n-1)/8 * (f1[i] - f1[i+1]);
    var vym{i in J}     =   (   vy[i] +    vy[i+1])/2 + tf/(n-1)/8 * (f2[i] - f2[i+1]);
    var zm{i in J}      =   (    z[i] +     z[i+1])/2 + tf/(n-1)/8 * (f3[i] - f3[i+1]);
    var vzm{i in J}     =   (   vz[i] +    vz[i+1])/2 + tf/(n-1)/8 * (f4[i] - f4[i+1]);
    var thetam{i in J}  =   (theta[i] + theta[i+1])/2 + tf/(n-1)/8 * (f5[i] - f5[i+1]);
    var omegam{i in J}  =   (omega[i] + omega[i+1])/2 + tf/(n-1)/8 * (f6[i] - f6[i+1]);
#-----------------------------------------------------------------------

#Dynamic at the mid-points----------------------------------------------
    var f1m{i in J} = vym[i];
    var f2m{i in J} =  -((urm[i]+ulm[i]) / mass * M + F / mass) * sin(thetam[i]) - beta * vym[i];
    var f3m{i in J} = vzm[i];
    var f4m{i in J} = ((urm[i]+ulm[i]) / mass * M + F / mass) * cos(thetam[i]) - gearth - beta * vzm[i];
    var f5m{i in J} = omegam[i];
    var f6m{i in J} = L / Ixx * M * (urm[i] - ulm[i]);

#-----------------------------------------------------------------------

#Simpson Formula---------------------------------------------------------
subject to
    dynamicx{i in J}:         y[i]  =    y[i+1] - tf/(n-1)/6*(f1[i] + f1[i+1] + 4*f1m[i]);
    dynamicvx{i in J}:       vy[i]  =   vy[i+1] - tf/(n-1)/6*(f2[i] + f2[i+1] + 4*f2m[i]);
    dynamicz{i in J}:         z[i]  =    z[i+1] - tf/(n-1)/6*(f3[i] + f3[i+1] + 4*f3m[i]);
    dynamicvz{i in J}:       vz[i] =    vz[i+1] - tf/(n-1)/6*(f4[i] + f4[i+1] + 4*f4m[i]);
    dynamictheta{i in J}: theta[i] = theta[i+1] - tf/(n-1)/6*(f5[i] + f5[i+1] + 4*f5m[i]);
    dynamicomega{i in J}: omega[i] = omega[i+1] - tf/(n-1)/6*(f6[i] + f6[i+1] + 4*f6m[i]);
#--------------------------------------------------------------------------

#Constraints------------------------------------------
    #Boundary Conditions
    subject to InitialPositionx:      y[1] = y0;
    subject to InitialPositionz:      z[1] = z0;
    subject to InitialVelocityx:     vy[1] = vy0;
    subject to InitialVelocityz:     vz[1] = vz0;
    subject to InitialPitch:      theta[1] = theta0;
    subject to InitialAngularVel: omega[1] = omega0;

    #Final
    subject to FinalPositionx:      y[n] = yn;
    subject to FinalPositionz:      z[n] = zn;
    subject to FinalVelocityz:     vz[n] = vzn;
    subject to FinalVelocityx:     vy[n] = vyn;
    subject to FinalPitch:      theta[n] = thetan;
    subject to FinalAngularVel: omega[n] = omegan;

    #Constraints on the control slope 
    #subject to ControlSlopel{i in J}:  abs( ul[i] - ulm[i]) / dt <= 1.;
    #subject to ControlSlopelm{i in J}: abs(ulm[i] -  ul[i+1]) / dt <= 1.;
    #subject to ControlSloper{i in J}:  abs( ur[i] - urm[i]) / dt <= 1.;
    #subject to ControlSloperm{i in J}: abs(urm[i] -  ur[i+1]) / dt <= 1.;
    
    
    # State constraint (theta)
    #subject to thetamagnitude1{i in I}: theta[i] - maxtheta <= 0;
    #subject to thetamagnitude2{i in I}: theta[i] + maxtheta >= 0;
    #subject to thetamagnitude1m{i in J}: thetam[i] - maxtheta <= 0;
    #subject to thetamagnitude2m{i in J}: thetam[i] + maxtheta >= 0;
#-------------------------------------------------------------

#Guess-------------------------------------------------------
    let tf := tn;
    let {i in I}  ul[i] := 0.5;
    let {i in I}  ur[i] := 0.5;
    let {i in J} ulm[i] := 0.5;
    let {i in J} urm[i] := 0.5;
#-------------------------------------------------------------

#Solver Options-----------------------------------------------
    option solver snopt;
    option substout 0;
    option show_stats 1;
    options snopt_options "outlev=2 Major_iterations=4000 Superbasics=1500";
#-------------------------------------------------------------

# #Solve!!!-----------------------------------------------------
  #solve;
# #-------------------------------------------------------------

# #Print the Solution with midpoints---------------------------
#     for {i in J}
#     {
#     printf "%17.16e %17.16e %17.16e %17.16e %17.16e %17.16e %17.16e %17.16e\n", timegrid[i],  x[i], vx[i], z[i], vz[i], theta[i], u1[i], u2[i] > out/sol_power.out;
#     printf "%17.16e %17.16e %17.16e %17.16e %17.16e %17.16e %17.16e %17.16e\n", timegrid[i] + dt/2, xm[i], vxm[i], zm[i], vzm[i], thetam[i], u1m[i], u2m[i] > out/sol_power.out;
#     }
#     printf "%17.16e %17.16e %17.16e %17.16e %17.16e %17.16e %17.16e %17.16e\n", timegrid[n],  x[n], vx[n], z[n], vz[n], theta[n], u1[n], u2[n] > out/sol_power.out;
#     close out/sol_power.out;
# #-------------------------------------------------------------
