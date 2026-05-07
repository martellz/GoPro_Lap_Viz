import numpy as np

def wgs84_to_ecef(latitude, longitude, height):
    latitude = np.array(latitude)
    longitude = np.array(longitude)
    height = np.array(height)
    s_lat = np.sin(latitude)
    c_lat = np.cos(latitude)
    s_long = np.sin(longitude)
    c_long = np.cos(longitude)

    N = 6378137 / np.sqrt(1.0 - 0.006694379990141108 * s_lat * s_lat)
    M = (N + height) * c_lat

    x = M * c_long
    y = M * s_long
    z = (N * (1.0 - 0.006694379990141108) + height) * s_lat
    return np.array([x,y,z])

def get_ecef_to_enu_transform(latitude, longitude, height):
    s_lat = np.sin(latitude)
    c_lat = np.cos(latitude)
    s_long = np.sin(longitude)
    c_long = np.cos(longitude)
    origin = wgs84_to_ecef(latitude, longitude, height)

    R = np.array(
        [
            -s_long,
            c_long,
            0,
            -s_lat * c_long,
            -s_lat * s_long,
            c_lat,
            c_lat * c_long,
            c_lat * s_long,
            s_lat,
        ]
    ).reshape((3, 3))
    t = -R @ origin
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3,3] = t
    return T

deg2rad = np.pi/180
rad2deg = 180/np.pi

origin_latitude = 30.5056 * deg2rad
origin_longitude = 114.413 * deg2rad

# T_ecef_to_enu = get_ecef_to_enu_transform(origin_latitude, origin_longitude, 17.74)

def wgs84_to_enu(latitude, longitude, height):
    T_ecef_to_enu = get_ecef_to_enu_transform(latitude[0], longitude[0], height[0])
    ecef = wgs84_to_ecef(latitude, longitude, height)
    enu = T_ecef_to_enu[0:3, 0:3] @ ecef + T_ecef_to_enu[0:3, 3:4]
    return enu