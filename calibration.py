

def calibrate(mode):
    """
    main function for score calibration. Depending on the mode, calls the appropirate function
    :param mode: "simple" or "recalibration_func"
    :return:
    """
    if mode == "simple":
        simple()
    elif mode == "recalibration_func":
        recalibration()

    # I should not be here
    raise ValueError("not implemented")

def simple():
    pass


def recalibration():
    pass