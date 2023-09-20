"""

Create observation data for a given object orbit which are observed by given satellites & ground stations.

"""


import numpy as np

from observations import *

def data(orbit, sat_orbits, station_ecefs, init_epoch_s):
    data = []
    for i in range(len(orbit.t)):
        PNRW, PN, T_UT1, dEps, omega_vec, R, W = ecef_eci(orbit.t[i]/86400, return_extras=True)
        ecef_eci_quants = [PNRW, PN, T_UT1, dEps, omega_vec, R, W]
        sun = sun_vector(T_UT1, PN)
        sun /= norm(sun)
        moon = moon_vector(T_UT1, PN, dEps)
        moon /= norm(moon)
        sat_states = [sat_orbit.y[:,i] for sat_orbit in sat_orbits]
        stufs = [sat_vis(orbit.t[i], orbit.y[:,i], sat_states[x], sun, moon) for x in range(len(sat_states))]
        sat_visibs = [stuf[0] for stuf in stufs]
        snrs = [stuf[1] for stuf in stufs]
        rad_visibs = [radar_vis(orbit.t[i], orbit.y[:,i], station_ecefs[x], ecef_eci_quants) for x in range(len(station_ecefs))]
        for j in range(len(sat_visibs)):
            if sat_visibs[j]:
                measurement = alt_and_dec(orbit.y[:,i], sat_states[j])
                angular_rate = ang_rate(orbit.y[:,i], sat_states[j])  # rad/s
                # ifov = 1.7e-4  # 10 deg FOV, 1000 pixels
                integration = 5  # s
                # num_pix = int(angular_rate * integration / ifov)
                ang_size = angular_rate * integration
                var = 100000 / snrs[j]
                err = np.random.normal(0, ang_size*var, 2)
                measurement =+ err
                R = np.diag(np.ones(2)*(ang_size*var)**2)
                data.append([0, j, orbit.t[i]-init_epoch_s, *measurement, *R.flatten(), *sat_states[j]])#*orbit.y[:,i]])
        for j in range(len(rad_visibs)):
            if rad_visibs[j]:
                measurement, station_state = range_and_rate(orbit.t[i], orbit.y[:,i], station_ecefs[j], return_extras=True)
                rel_power_loss = (measurement[0]/1e6)**4
                # rel_power_loss = (measurement[0]/1e9)**4
                err_range = np.random.normal(0, 10*rel_power_loss)
                err_rate = np.random.normal(0, 0.005*rel_power_loss)
                err = np.array([err_range, err_rate])
                measurement += err
                R = np.diag(np.ones(2)*np.array([(10*rel_power_loss)**2, (0.005*rel_power_loss)**2]))
                data.append([1, j, orbit.t[i]-init_epoch_s, *measurement, *R.flatten(), *station_state])
    data = np.array(data)
    if len(data)==0:
        print("Warning: no observations taken")
        raise Exception
    if not np.any(data[:,0] == 0):
        print("Warning: no optical observations")
    if not np.any(data[:,0] == 1):
        print("Warning: no radar observations")
    return data

def write_data(data):
    if len(data)==0:
        print("Warning: no observations taken")
        raise Exception
    np.savetxt("./code_data_files/observations.txt", data)

def read_data(data):
    return np.laodtxt("./code_data_files/observations.txt")
