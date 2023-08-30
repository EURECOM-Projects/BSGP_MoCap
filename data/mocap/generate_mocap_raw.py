import os
import numpy as np

# data root contains a subfolder "mocap/raw" with amc files from CMU MOCAP databse : http://mocap.cs.cmu.edu/
# directory structure: DATA_ROOT/mocap/raw/subjects/*subject_id/*action.amc
DATA_ROOT = './'


# stolen as-is from https://github.com/CalciferZh/AMCParser
def read_line(stream, idx):
    if idx >= len(stream):
        return None, idx
    line = stream[idx].strip().split()
    idx += 1
    return line, idx


def parse_amc(file_path):
    with open(file_path) as f:
        content = f.read().splitlines()

    for idx, line in enumerate(content):
        if line == ':DEGREES':
            content = content[idx + 1:]
            break

    frames = []
    idx = 0
    line, idx = read_line(content, idx)
    assert line[0].isnumeric(), line
    EOF = False
    while not EOF:
        joint_degree = {}
        while True:
            line, idx = read_line(content, idx)
            if line is None:
                EOF = True
                break
            if line[0].isnumeric():
                break
            joint_degree[line[0]] = [float(deg) for deg in line[1:]]
        frames.append(joint_degree)
    return frames


def amc_to_mat(motions):
    readings = []
    for motion in motions:
        locations = [0, 6, 9, 12, 15, 18, 21, 24, 26, 29, 30, 31, 33, 34, 36, 38, 41, 42, 43, 45, 46, 48, 51, 52, 54,
                     55, 58, 59, 61]
        joint_readings = list([None] * 62)
        for (joint_name, joint_vals), joint_idx in zip(motion.items(), locations):
            for j, val in enumerate(joint_vals):
                joint_readings[joint_idx + j] = val
        readings.append(joint_readings)
    return np.stack(readings)


def load_mocap_data(subject, actions, startframes=0, skipframes=1, endframes=None):
    mask = [31, 32, 33, 34, 35, 43, 44, 45, 46, 47, 54, 61]
    # mask: rhand, rfingers, rthumb, lhand, lfingers, lthumb, rtoes, ltoes
    Ys = []
    for action in actions:
        filename = os.path.join(DATA_ROOT,'subjects/{:02d}/{:02d}_{:02d}.amc'.format(subject, subject, action))
        try:
            motions = parse_amc(filename)
        except FileNotFoundError:
            print("download and extract data from http://mocap.cs.cmu.edu/!")
            raise
        motions_mat = amc_to_mat(motions) # 62 features (complete)

        Y = np.delete(motions_mat, mask, 1)
        Y = Y[startframes:endframes:skipframes, :]

        globY = Y[1:, :3] - Y[:-1, :3]
        globY = np.concatenate([globY, globY[-1:]], 0)
        Y[:, :3] = globY

        Ys.append(Y)

    Ys = np.stack(Ys)
    Ys = Ys - Ys.mean((0, 1))

    return Ys

# SUBJECTS

def create_mocap07():
    subject = 7
    actions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    startframes = 0
    endframes = 300
    skipframes = 1

    Y = load_mocap_data(subject, actions, startframes, skipframes, endframes)
    np.savez(os.path.join(DATA_ROOT, 'mocap07.npz'), data=Y)

def create_mocap08():
    subject = 8
    actions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    startframes = 0
    endframes = 270
    skipframes = 1

    Y = load_mocap_data(subject, actions, startframes, skipframes, endframes)
    np.savez(os.path.join(DATA_ROOT, 'mocap08.npz'), data=Y)

def create_mocap09():
    subject = 9
    actions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11]
    startframes = 0
    endframes = 120
    skipframes = 1

    Y = load_mocap_data(subject, actions, startframes, skipframes, endframes)
    np.savez(os.path.join(DATA_ROOT, 'mocap09.npz'), data=Y)

def create_mocap16():
    subject = 16
    actions = [35, 36, 45, 46, 55, 56]
    startframes = 0
    endframes = 120
    skipframes = 1

    Y = load_mocap_data(subject, actions, startframes, skipframes, endframes)
    np.savez(os.path.join(DATA_ROOT, 'mocap16.npz'), data=Y)

def create_mocap35():
    subject = 35
    actions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 17, 28, 19, 20, 21, 22, 23, 24, 25, 26]
    startframes = 0
    endframes = 300
    skipframes = 1

    Y = load_mocap_data(subject, actions, startframes, skipframes, endframes)

    Y_train = Y[:16]
    Y_valid = Y[16:19]
    Y_test = Y[19:23]
    np.savez(os.path.join(DATA_ROOT, 'mocap35.npz'), test=Y_test, validation=Y_valid, train=Y_train)


def create_mocap39():
    subject = 39
    actions = [1, 2, 7, 8, 9, 10, 3, 4, 5, 6]  # ,12,13,14]
    startframes = 0
    endframes = 300
    skipframes = 1

    Y = load_mocap_data(subject, actions, startframes, skipframes, endframes)

    Y_train = Y[:6]
    Y_valid = Y[6:8]
    Y_test = Y[8:10]
    np.savez(os.path.join(DATA_ROOT, 'mocap39.npz'), test=Y_test, validation=Y_valid, train=Y_train)

create_mocap07()
create_mocap08()
create_mocap09()
create_mocap16()
