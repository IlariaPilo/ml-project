import numpy as np

def convert_txt_to_np(file_name):
    """
    Loads the content of a file in a pair X (dataset) and L (labels), then saves them as npy files (to be used only once per file).
    :param file_name is the name of the file (WITHOUT EXTENSION)
    """
    # open the file in read mode
    fp = open(file_name+'.txt', 'r')
    X = []
    L = []
    # read it line by line
    while True:
        line = fp.readline()
        if not line:
            break
        # split the line
        clean = [float(x.strip()) for x in line.split(',')]
        X.append(clean[0:-1])
        L.append(clean[-1])
    fp.close()
    X, L = np.array(X).T, np.array(L, dtype=int)
    np.save(file_name+'X.npy', X)
    np.save(file_name+'L.npy', L)


def load(file_name):
    """
    Load the content of the pair file_name+'X.npy' and file_name+'L.npy'.
    :param file_name is the name of the file (WITHOUT EXTENSION)
    """
    return np.load(file_name+'X.npy'), np.load(file_name+'L.npy')

if __name__ == '__main__':
    # convert from txt to binary
    convert_txt_to_np("data/Train")
    convert_txt_to_np("data/Test")
